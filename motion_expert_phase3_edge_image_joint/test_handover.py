"""Owned handover admission and directional image-joint checkpoint contracts."""
import copy
import os
from unittest.mock import patch
import pytest
from motion_expert_phase3_edge_joint.settings import RECIPE
from . import launch
from .policy import configure, check_checkpoint

def test_uncoordinated_default_still_requires_empty_gpus():
    with patch.object(launch,"require_empty") as check:
        launch.require_available(["gpu0"],{})
        check.assert_called_once_with(["gpu0"])

def test_coordinated_set_is_exact_and_own_processes_only():
    with patch.object(launch,"resident_pids",side_effect=[{11},{12}]),patch.object(launch,"process_identity",return_value=(os.getuid(),"start")):
        residents=launch.coordinated_residents(["gpu0","gpu1"],"11,12")
    assert residents=={"gpu0":{11:(os.getuid(),"start")},"gpu1":{12:(os.getuid(),"start")}}
    with patch.object(launch,"resident_pids",return_value={11}),pytest.raises(RuntimeError,match="exactly match"):
        launch.coordinated_residents(["gpu0"],"11,12")
    with patch.object(launch,"resident_pids",return_value={11}),patch.object(launch,"process_identity",return_value=(os.getuid()+1,"start")),pytest.raises(RuntimeError,match="this user's"):
        launch.coordinated_residents(["gpu0"],"11")

def test_sharing_rejects_new_or_reused_pids_and_low_headroom():
    residents={"gpu0":{11:(os.getuid(),"start")}}
    with patch.object(launch,"resident_pids",return_value={12}),pytest.raises(RuntimeError,match="Uncoordinated"):
        launch.require_available(["gpu0"],residents)
    with patch.object(launch,"resident_pids",return_value={11}),patch.object(launch,"process_identity",return_value=(os.getuid(),"reused")),pytest.raises(RuntimeError,match="replaced"):
        launch.require_available(["gpu0"],residents)
    with patch.object(launch,"resident_pids",return_value={11}),patch.object(launch,"process_identity",return_value=(os.getuid(),"start")),patch.object(launch.subprocess,"check_output",return_value="60000"),pytest.raises(RuntimeError,match="64GiB"):
        launch.require_available(["gpu0"],residents)

def test_departed_old_process_is_allowed_but_no_new_process():
    with patch.object(launch,"resident_pids",return_value=set()),patch.object(launch.subprocess,"check_output",return_value="120000"):
        launch.require_available(["gpu0"],{"gpu0":{11:(os.getuid(),"start")}})


def test_two_task_entrypoint_attention_survives_shared_launcher(monkeypatch):
    from motion_expert_phase3_edge_mmdit.policy import configure as configure_two_task
    previous=copy.deepcopy(RECIPE)
    try:
        monkeypatch.setenv("PHASE3_MMDIT_CONDITIONING_ATTENTION","bidirectional")
        monkeypatch.setenv("PHASE3_IMAGE_JOINT_CONDITIONING_ATTENTION","directional")
        assert launch.configure_attention_policy("motion_expert_phase3_edge_mmdit",None)=="bidirectional"
        assert configure_two_task()["conditioning_attention"]=="bidirectional"
        assert RECIPE["tasks"]=={"video2motion":.5,"motimg2video":.5}
        assert launch.configure_attention_policy("motion_expert_phase3_edge_mmdit","directional")=="directional"
        assert configure_two_task()["conditioning_attention"]=="directional"
        assert RECIPE["tasks"]=={"video2motion":.5,"motimg2video":.5}
    finally:
        RECIPE.clear();RECIPE.update(previous)

@pytest.mark.parametrize("attention",["directional","bidirectional"])
def test_joint_policy_roundtrip_and_worker_initialization(attention,monkeypatch):
    previous=copy.deepcopy(RECIPE)
    try:
        monkeypatch.setenv("PHASE3_IMAGE_JOINT_CONDITIONING_ATTENTION",attention)
        recipe=copy.deepcopy(configure())
        assert recipe["conditioning_attention"]==attention
        assert recipe["tasks"]=={"video2motion":.4,"motimg2video":.4,"joint":.2}
        assert recipe["joint_conditioning"]=="first_image_only_plus_supplied_neutral_shape_no_caption"
        # A checkpoint restores its own direction even in an opposite ambient env.
        monkeypatch.setenv("PHASE3_IMAGE_JOINT_CONDITIONING_ATTENTION","bidirectional" if attention=="directional" else "directional")
        check_checkpoint({"scheduler":{"recipe":recipe}})
        assert RECIPE["conditioning_attention"]==attention
        from .data import Phase3Dataset
        with patch("motion_expert_phase3_edge_image_joint.train.configure_local_paths"),patch.dict(os.environ,{"PHASE3_IMAGE_JOINT_CONDITIONING_ATTENTION":attention}):
            worker=Phase3Dataset.__new__(Phase3Dataset);worker.__setstate__({})
        assert RECIPE["conditioning_attention"]==attention and RECIPE["tasks"]==recipe["tasks"]
        bad=copy.deepcopy(recipe);bad["conditioning_attention"]="bidirectional" if attention=="directional" else "directional"
        with pytest.raises(ValueError,match="conditioning_attention"):
            check_checkpoint({"scheduler":{"recipe":bad}})
    finally:
        RECIPE.clear();RECIPE.update(previous)

"""Perceptually smooth, fixed-range indigo-to-muted-rose error color map."""
import numpy as np

LOW='#7B77FF'
HIGH='#D8768C'


def srgb_to_linear(value):
    x=np.asarray(value,dtype=np.float64)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def linear_to_srgb(value):
    x=np.asarray(value,dtype=np.float64)
    return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)


def hex_rgb(value):
    return np.array([int(value[i:i+2],16)/255 for i in (1,3,5)])


def rgb_to_lab(rgb):
    rgb=srgb_to_linear(rgb)
    lms=rgb@np.array([[.4122214708,.2119034982,.0883024619],
        [.5363325363,.6806995451,.2817188376],[.0514459929,.1073969566,.6299787005]])
    return np.cbrt(lms)@np.array([[.2104542553,1.9779984951,.0259040371],
        [.7936177850,-2.4285922050,.7827717662],[-.0040720468,.4505937099,-.8086757660]])


def lab_to_rgb(lab):
    lms=lab@np.array([[1.,1.,1.],[.3963377774,-.1055613458,-.0894841775],
        [.2158037573,-.0638541728,-1.2914855480]])
    rgb=lms**3@np.array([[4.0767416621,-1.2684380046,-.0041960863],
        [-3.3077115913,2.6097574011,-.7034186147],[.2309699292,-.3413193965,1.7076147010]])
    return np.clip(linear_to_srgb(rgb),0,1)


def error_colors(errors_m,maximum_cm=30.,low=LOW,high=HIGH):
    e=np.asarray(errors_m,dtype=np.float64)
    if not np.isfinite(e).all() or np.any(e<0) or maximum_cm<=0:
        raise ValueError('Finite nonnegative metric errors and positive fixed range required')
    mix=np.clip(e*100/maximum_cm,0,1)[...,None]
    start,end=rgb_to_lab(hex_rgb(low)),rgb_to_lab(hex_rgb(high))
    colors=lab_to_rgb(start*(1-mix)+end*mix)
    # Endpoints must equal the paper palette exactly, not a matrix-roundoff tint.
    return np.where(mix==0,hex_rgb(low),np.where(mix==1,hex_rgb(high),colors)).astype(np.float32)

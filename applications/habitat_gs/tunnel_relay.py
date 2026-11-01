"""Optional loopback-only login-host TCP relay; performs no rendering or ML."""
import argparse
import select
import socket
import socketserver


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node',required=True)
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--target-port',type=int,default=8765)
    args=p.parse_args()
    class Relay(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                with socket.create_connection((args.node,args.target_port),timeout=10) as target:
                    target.settimeout(None)
                    sockets=[self.request,target]
                    while True:
                        readable,_,_=select.select(sockets,[],[],120)
                        if not readable:return
                        for source in readable:
                            data=source.recv(65536)
                            if not data:return
                            (target if source is self.request else self.request).sendall(data)
            except (OSError,ConnectionError):return
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address=True
        daemon_threads=True
    with Server(('127.0.0.1',args.port),Relay) as server:
        print(f'Loopback relay 127.0.0.1:{args.port} -> {args.node}:{args.target_port}',flush=True)
        server.serve_forever()


if __name__=='__main__':main()

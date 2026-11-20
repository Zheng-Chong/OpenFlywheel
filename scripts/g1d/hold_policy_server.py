"""A stand-in G1-D policy server that holds the robot still: it answers every request of scripts/g1d/remote_policy.py
with its own joint16 state repeated over the horizon. Speaks the same openpi msgpack WebSocket protocol as a real
model, so it checks the evaluation pipeline (assets, server start, simulations, scoring, publishing) without a GPU
model; every episode fails the task.

    robodojo-sim python scripts/g1d/hold_policy_server.py --port 8001 [--horizon 16]
As a model version's serve command on the platform (which asks for {ckpt}; this server reads no weights):
    robodojo-sim python scripts/g1d/hold_policy_server.py --ckpt {ckpt} --port {port}
"""

import argparse
import asyncio

import numpy as np

from scripts.g1d.remote_policy import codec


async def serve(port, horizon):
    from websockets import serve as ws_serve

    pack, unpack = codec()

    async def handler(ws):
        await ws.send(pack({"policy": "hold", "action_horizon": horizon}))
        async for message in ws:
            state = np.asarray(unpack(message)["observation/state"], np.float32).reshape(16)
            await ws.send(pack({"actions": np.repeat(state[None], horizon, axis=0)}))

    async with ws_serve(handler, "0.0.0.0", port, max_size=None, compression=None):
        print(f"[hold] listening on {port}", flush=True)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--ckpt", help="ignored")
    args = parser.parse_args()
    asyncio.run(serve(args.port, args.horizon))


if __name__ == "__main__":
    main()

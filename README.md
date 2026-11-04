# Holo: a live AI guide that sees what you see

Say "Muse, how do I…" and Holo highlights the real parts in front of you, step by step. It advances when it sees each step happen, and it answers mid-task questions with web search. In **Build mode**, "Muse, help me build a small FPV drone" gives you:

1. A design: parts, printable parts, wiring, code and steps.
2. A rendered image of the finished build.
3. A 3D model you can hold with your hands and export as an STL to print.

## Stack
- **Muse Spark Open** (Meta Model API via Baseten), in `server.py`. It plans from the camera frame, names step targets, routes questions, runs agentic web search and designs builds.
- **SAM 3.1** handles text-prompted video segmentation and tracking of every part Muse names, plus live single-frame highlights (`perception/pod/sam31_server.py`, `perception/pod/auto_pod.py`).
- **DINOv3 ViT-S/16** provides dense patch features pooled inside masks, used for the identity of look-alike parts.
- **Build generation** (`perception/pod/gen_server.py`): Z-Image-Turbo (image) and Hunyuan3D-2mini (image → watertight GLB/STL).
- **Unity app** (`unity/`): Meta XR Interaction SDK and XR Hands, a Horizon OS–style glass UI, wake word "Muse", local Whisper STT and Piper TTS, SAM 3.1 highlights, and a 3D model viewer (pinch to grab, two hands to scale and rotate, follow/pin, top view, STL export).
- **Automatic guide** (`perception/auto_render.py`): a step state machine driven only by Muse's plan and the masks. Nothing is specific to one clip.

## Run
1. GPU box:
   - `uvicorn sam31_server:app --port 8790`
   - `uvicorn gen_server:app --port 8791`
   - Tunnel both ports to the laptop.
2. Laptop: put `meta_baseten=<key>` in `.env`, then run `uvicorn server:app --port 8765`.
3. Unity 6: add `unity/` under `Assets/Holo`, then run *Holo > Build App Scene* and press Play.

Demo footage: Mixkit free stock clips (Mixkit licence).

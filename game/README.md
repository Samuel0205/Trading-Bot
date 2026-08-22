# Operation Iron Talon — top-down military strike prototype

A single-file, no-dependency HTML5 game inspired by the mobile-gunship screenshot:
protect the **friendly (green)**, follow the **red markers** to hostiles, and
eliminate them wave after wave.

## Play it

Open `game/index.html` in any modern browser. No build step, no assets, no server.

```
# from the repo root
open game/index.html        # macOS
xdg-open game/index.html    # Linux
# or just double-click the file
```

## Controls

| Action  | Keyboard / Mouse            | Touch                    |
|---------|-----------------------------|--------------------------|
| Move    | `W A S D` / arrow keys      | drag on the battlefield  |
| Aim     | mouse cursor                | drag direction           |
| Fire    | click / hold, or `FIRE` btn | hold the **FIRE** button |
| Airstrike | `Q` or the **AIRSTRIKE** button (costs 10 ⚡) | tap **AIRSTRIKE** |

## The loop

- **Mission briefing** modal (the `AFFIRMATIVE` screen) opens each wave.
- A **green friendly APC** crawls a patrol route. Keep its integrity above 0%.
- **Red hostiles** spawn from the edges and attack the friendly (and you).
  Off-screen ones show as **red HUD arrows**; the friendly shows as a **green arrow**.
- Kills pay **cash**; some drop **⚡ energy** / **$** pickups; occasional **gold**.
- Clear all hostiles to bank a bonus and advance. Waves get bigger and tougher.
- Friendly integrity carries between waves. If it hits 0%, mission failed → redeploy.

## What this is (and isn't)

This is a **playable prototype of the game *loop and HUD*** from the reference
image — HUD bar (energy / cash / gold), briefing modal, protect-the-green,
hunt-the-red, waves and rewards — rendered entirely with procedural Canvas 2D
graphics so it runs anywhere with zero assets.

It is **not** a 3D photoreal title. Matching that screenshot's fidelity means a
real engine (Unity/Unreal), 3D models, and licensed art — an art-team effort, not
a single file. This prototype is the honest, runnable first step: nail the
gameplay feel, then re-skin it in an engine if you want to take it further.

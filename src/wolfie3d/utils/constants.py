#!/usr/bin/env python3
"""
Constants for Wolfie3D game
"""

import math

# ---------- Konfig ----------
WIDTH, HEIGHT = 1024, 600
HALF_W, HALF_H = WIDTH // 2, HEIGHT // 2
FPS = 60

# Kamera/FOV
FOV = 66 * math.pi / 180.0
PLANE_LEN = math.tan(FOV / 2)

# Bevegelse
MOVE_SPEED = 3.0      # enheter/sek
ROT_SPEED = 2.0       # rad/sek
STRAFE_SPEED = 2.5

# Våpen
WEAPON_PISTOL = 1
WEAPON_SAU = 2

# Tekstur-størrelse brukt på GPU (proseduralt generert)
TEX_W = TEX_H = 256

# Depth mapping (lineær til [0..1] for gl_FragDepth)
FAR_PLANE = 100.0

# Kart (0=tomt, >0=veggtype/tekstur-id)
MAP: list[list[int]] = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,2,2,2,2,2,0,0,0,0,3,3,3,0,0,4,4,4,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]
MAP_W = len(MAP[0])
MAP_H = len(MAP)

# Startpos og retning
PLAYER_START_X = 3.5
PLAYER_START_Y = 10.5
PLAYER_START_DIR_X = 1.0
PLAYER_START_DIR_Y = 0.0
PLAYER_START_PLANE_X = 0.0
PLAYER_START_PLANE_Y = PLANE_LEN

# Spiller HP
PLAYER_MAX_HP = 150  # Økt fra 100 til 150
PLAYER_INVULNERABLE_TIME = 0.5  # Sekunder hvor spilleren ikke kan ta skade

#!/usr/bin/env python3
"""
Vibe Wolf (Python + PyOpenGL) — GL-renderer
-------------------------------------------
Denne varianten beholder logikken (kart, DDA-raycasting, input, sprites),
men tegner ALT med OpenGL (GPU). Vegger og sprites blir teksturerte quads,
og vi bruker depth-test i GPU for korrekt okklusjon (ingen CPU zbuffer).

Avhengigheter:
  - pygame >= 2.1 (for vindu/input)
  - PyOpenGL, PyOpenGL-accelerate
  - numpy

Kjør:
  python wolfie3d_gl.py

Taster:
  - WASD / piltaster: bevegelse
  - Q/E eller ← → : rotasjon
  - SPACE / venstre mus: skyte
  - 1: bytt til pistol
  - 2: bytt til sau
  - ESC: avslutt
"""

from __future__ import annotations

import math
import sys
import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pygame
from OpenGL import GL as gl

# Import refactored modules
from .utils.constants import *
from .utils.input import handle_input
from .world.map import try_move, in_map, is_wall, clamp01, find_valid_spawn_position
from .entities.bullet import Bullet, SauBullet
from .entities.enemy import Enemy

if TYPE_CHECKING:  # kun for typing hints
    from collections.abc import Sequence

# ---------- Font håndtering ----------
def get_font(size: int) -> pygame.font.Font:
    """Henter en font som alltid fungerer på macOS og andre systemer.
    
    Bruker pygame's innebygde bitmap-font som er garantert tilgjengelig.
    Dette løser problemet med tekst som rendres som firkanter.
    """
    pygame.font.init()
    
    # Prøv først pygame's innebygde freesans font som alltid er tilgjengelig
    try:
        font_path = pygame.font.match_font('freesans')
        if font_path:
            font = pygame.font.Font(font_path, size)
            test_surface = font.render("A", True, (255, 255, 255))
            if test_surface.get_width() > 0 and test_surface.get_height() > 0:
                print("[Font] Bruker pygame's innebygde freesans font")
                return font
    except:
        pass
    
    # Fallback til system default
    try:
        font = pygame.font.Font(None, size)
        test_surface = font.render("A", True, (255, 255, 255))
        if test_surface.get_width() > 0 and test_surface.get_height() > 0:
            print("[Font] Bruker system default font")
            return font
    except:
        pass
    
    # Siste utvei: bruk pygame's fallback font
    print("[Font] Bruker pygame's fallback font")
    return pygame.font.Font(None, size)

# ---------- OpenGL tekstrendering ----------
_text_cache = {}  # (string, size, color) -> (tex_id, w, h)

def clear_text_cache():
    """Rydder opp i tekst-cachen ved å frigjøre OpenGL teksturer."""
    global _text_cache
    for tex_id, w, h in _text_cache.values():
        try:
            gl.glDeleteTextures([tex_id])
        except:
            pass
    _text_cache.clear()

def draw_text_px(renderer, text: str, x: int, y: int, size: int = 32, color: tuple = (255, 255, 255)) -> None:
    """Tegner tekst via OpenGL som tekstur i stedet for pygame blit.
    
    Dette løser problemet med tekst som rendres som firkanter på macOS
    når man bruker pygame.OPENGL mode.
    """
    key = (text, size, color)
    
    # Cache tekstur hvis den ikke allerede eksisterer
    if key not in _text_cache:
        font = get_font(size)
        surf = font.render(text, True, color)
        tex_id = surface_to_texture(surf)
        w, h = surf.get_width(), surf.get_height()
        _text_cache[key] = (tex_id, w, h)
    
    tex_id, w, h = _text_cache[key]
    
    # Bygg skjerm-space quad i piksler (topleft = (x,y))
    x0 = (2.0 * x) / WIDTH - 1.0
    x1 = (2.0 * (x + w)) / WIDTH - 1.0
    y0 = 1.0 - 2.0 * (y / HEIGHT)
    y1 = 1.0 - 2.0 * ((y + h) / HEIGHT)
    
    r = g = b = 1.0
    depth = 0.0  # helt foran
    verts = np.asarray([
        x0, y0, 0.0, 1.0, r, g, b, depth,  # Bottom-left
        x0, y1, 0.0, 0.0, r, g, b, depth,  # Top-left
        x1, y0, 1.0, 1.0, r, g, b, depth,  # Bottom-right
        x1, y0, 1.0, 1.0, r, g, b, depth,  # Bottom-right
        x0, y1, 0.0, 0.0, r, g, b, depth,  # Top-left
        x1, y1, 1.0, 0.0, r, g, b, depth,  # Top-right
    ], dtype=np.float32).reshape((-1, 8))
    
    renderer.draw_arrays(verts, tex_id, use_tex=True)

# ---------- Spiller tilstand ----------
current_weapon = WEAPON_PISTOL

# Startpos og retning
player_x = PLAYER_START_X
player_y = PLAYER_START_Y
dir_x, dir_y = PLAYER_START_DIR_X, PLAYER_START_DIR_Y
plane_x, plane_y = PLAYER_START_PLANE_X, PLAYER_START_PLANE_Y

# Spiller HP
player_hp = PLAYER_MAX_HP
player_max_hp = PLAYER_MAX_HP
player_invulnerable_time = 0.0  # Tid hvor spilleren ikke kan ta skade
player_damage_flash_time = 0.0  # Tid for rød flash-effekt når spilleren tar skade

# Score system
player_score = 0

# Sau system
sau_count = 5  # Begrenset til 5 sauer - hver sau gjør 2 skade
active_sau = None  # Aktive sau som kan eksplodere
explosion_time = 0.0  # Tid for eksplosjons-effekt
explosion_x = 0.0  # X-posisjon for eksplosjon
explosion_y = 0.0  # Y-posisjon for eksplosjon

# ---------- Highscore system ----------
def load_highscores() -> list[dict]:
    """Laster highscores fra fil."""
    try:
        if os.path.exists("highscores.json"):
            with open("highscores.json", "r") as f:
                return json.load(f)
    except:
        pass
    return []

def save_highscores(highscores: list[dict]) -> None:
    """Lagrer highscores til fil."""
    try:
        with open("highscores.json", "w") as f:
            json.dump(highscores, f, indent=2)
    except:
        pass

def is_highscore(score: int) -> bool:
    """Sjekker om score er en highscore."""
    highscores = load_highscores()
    if len(highscores) < 10:  # Mindre enn 10 highscores
        return True
    return score > min(hs["score"] for hs in highscores)

def add_highscore(name: str, score: int) -> None:
    """Legger til en ny highscore."""
    highscores = load_highscores()
    highscores.append({"name": name, "score": score})
    # Sorter etter score (høyest først)
    highscores.sort(key=lambda x: x["score"], reverse=True)
    # Behold kun top 10
    highscores = highscores[:10]
    save_highscores(highscores)

# ---------- Hjelpere ----------
# ---------- Prosedural tekstur (pygame.Surface) ----------


# ---------- Prosedural tekstur (pygame.Surface) ----------
def make_brick_texture() -> pygame.Surface:
    surf = pygame.Surface((TEX_W, TEX_H))
    surf.fill((150, 40, 40))
    mortar = (200, 200, 200)
    brick_h = TEX_H // 4
    brick_w = TEX_W // 4
    for row in range(0, TEX_H, brick_h):
        offset = 0 if (row // brick_h) % 2 == 0 else brick_w // 2
        for col in range(0, TEX_W, brick_w):
            rect = pygame.Rect((col + offset) % TEX_W, row, brick_w - 1, brick_h - 1)
            pygame.draw.rect(surf, (165, 52, 52), rect)
    for y in range(0, TEX_H, brick_h):
        pygame.draw.line(surf, mortar, (0, y), (TEX_W, y))
    for x in range(0, TEX_W, brick_w):
        pygame.draw.line(surf, mortar, (x, 0), (x, TEX_H))
    return surf

def make_stone_texture() -> pygame.Surface:
    surf = pygame.Surface((TEX_W, TEX_H))
    base = (110, 110, 120)
    surf.fill(base)
    for y in range(TEX_H):
        for x in range(TEX_W):
            if ((x * 13 + y * 7) ^ (x * 3 - y * 5)) & 15 == 0:
                c = 90 + ((x * y) % 40)
                surf.set_at((x, y), (c, c, c))
    for i in range(5):
        pygame.draw.line(surf, (80, 80, 85), (i*12, 0), (TEX_W-1, TEX_H-1 - i*6), 1)
    return surf

def make_wood_texture() -> pygame.Surface:
    surf = pygame.Surface((TEX_W, TEX_H))
    for y in range(TEX_H):
        for x in range(TEX_W):
            v = int(120 + 40 * math.sin((x + y*0.5) * 0.12) + 20 * math.sin(y * 0.3))
            v = max(60, min(200, v))
            surf.set_at((x, y), (140, v, 60))
    for x in range(0, TEX_W, TEX_W // 4):
        pygame.draw.line(surf, (90, 60, 30), (x, 0), (x, TEX_H))
    return surf

def make_metal_texture() -> pygame.Surface:
    surf = pygame.Surface((TEX_W, TEX_H), pygame.SRCALPHA)
    base = (140, 145, 150, 255)
    surf.fill(base)
    for y in range(8, TEX_H, 16):
        for x in range(8, TEX_W, 16):
            pygame.draw.circle(surf, (90, 95, 100, 255), (x, y), 2)
    for y in range(TEX_H):
        shade = 130 + (y % 8) * 2
        pygame.draw.line(surf, (shade, shade, shade+5, 255), (0, y), (TEX_W, y), 1)
    return surf

def make_bullet_texture() -> pygame.Surface:
    surf = pygame.Surface((32, 32), pygame.SRCALPHA)
    pygame.draw.circle(surf, (255, 240, 150, 220), (16, 16), 8)
    pygame.draw.circle(surf, (255, 255, 255, 255), (13, 13), 3)
    return surf

# ---------- OpenGL utils ----------
VS_SRC = """
#version 330 core
layout (location = 0) in vec2 in_pos;    // NDC -1..1
layout (location = 1) in vec2 in_uv;
layout (location = 2) in vec3 in_col;    // per-vertex farge (for dimming/overlay)
layout (location = 3) in float in_depth; // 0..1 depth (0 nær, 1 fjern)

out vec2 v_uv;
out vec3 v_col;
out float v_depth;

void main() {
    v_uv = in_uv;
    v_col = in_col;
    v_depth = in_depth;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FS_SRC = """
#version 330 core
in vec2 v_uv;
in vec3 v_col;
in float v_depth;

out vec4 fragColor;

uniform sampler2D uTexture;
uniform bool uUseTexture;

void main() {
    vec4 base = vec4(1.0);
    if (uUseTexture) {
        base = texture(uTexture, v_uv);
        if (base.a < 0.01) discard; // alpha for sprites
    }
    vec3 rgb = base.rgb * v_col;
    fragColor = vec4(rgb, base.a);
    // Skriv eksplisitt dybde (lineær i [0..1])
    gl_FragDepth = clamp(v_depth, 0.0, 1.0);
}
"""

def compile_shader(src: str, stage: int) -> int:
    sid = gl.glCreateShader(stage)
    gl.glShaderSource(sid, src)
    gl.glCompileShader(sid)
    status = gl.glGetShaderiv(sid, gl.GL_COMPILE_STATUS)
    if status != gl.GL_TRUE:
        log = gl.glGetShaderInfoLog(sid).decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return sid

def make_program(vs_src: str, fs_src: str) -> int:
    vs = compile_shader(vs_src, gl.GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    ok = gl.glGetProgramiv(prog, gl.GL_LINK_STATUS)
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    if ok != gl.GL_TRUE:
        log = gl.glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"Program link error:\n{log}")
    return prog

def surface_to_texture(surf: pygame.Surface) -> int:
    """Laster pygame.Surface til GL_TEXTURE_2D (RGBA8). Returnerer texture id."""
    data = pygame.image.tostring(surf.convert_alpha(), "RGBA", True)
    w, h = surf.get_width(), surf.get_height()
    tid = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tid)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tid

def make_white_texture() -> int:
    surf = pygame.Surface((1, 1), pygame.SRCALPHA)
    surf.fill((255, 255, 255, 255))
    return surface_to_texture(surf)

def make_enemy_texture() -> pygame.Surface:
    s = pygame.Surface((256, 256), pygame.SRCALPHA)
    # kropp
    pygame.draw.rect(s, (60, 60, 70, 255), (100, 80, 56, 120), border_radius=6)
    # hode
    pygame.draw.circle(s, (220, 200, 180, 255), (128, 70), 26)
    # hjelm-ish
    pygame.draw.arc(s, (40, 40, 50, 255), (92, 40, 72, 40), 3.14, 0, 6)
    # "arm"
    pygame.draw.rect(s, (60, 60, 70, 255), (86, 110, 24, 16))
    pygame.draw.rect(s, (60, 60, 70, 255), (146, 110, 24, 16))
    return s


# ---------- GL Renderer state ----------
from pathlib import Path
import os
import pygame
from OpenGL import GL as gl

# ---------- GL Renderer state ----------
class GLRenderer:
    def __init__(self) -> None:
        # Shader program
        self.prog = make_program(VS_SRC, FS_SRC)
        gl.glUseProgram(self.prog)
        self.uni_tex = gl.glGetUniformLocation(self.prog, "uTexture")
        self.uni_use_tex = gl.glGetUniformLocation(self.prog, "uUseTexture")
        gl.glUniform1i(self.uni_tex, 0)

        # VAO/VBO (dynamisk buffer per draw)
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        stride = 8 * 4  # 8 float32 per vertex
        # in_pos (loc 0): 2 floats
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(0))
        # in_uv (loc 1): 2 floats
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(2 * 4))
        # in_col (loc 2): 3 floats
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(4 * 4))
        # in_depth (loc 3): 1 float
        gl.glEnableVertexAttribArray(3)
        gl.glVertexAttribPointer(3, 1, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(7 * 4))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        # Teksturer
        self.white_tex = make_white_texture()
        self.textures: dict[int, int] = {}  # tex_id -> GL texture

        # Last fra assets hvis tilgjengelig, ellers fall tilbake til proseduralt
        self.load_textures()

        # GL state
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)

    # ---------- teksturhjelpere ----------
    @staticmethod
    def _scale_if_needed(surf: pygame.Surface, size: int = 512) -> pygame.Surface:
        if surf.get_width() != size or surf.get_height() != size:
            surf = pygame.transform.smoothscale(surf, (size, size))
        return surf

    def _load_texture_file(self, path: str, size: int = 512) -> int:
        surf = pygame.image.load(path).convert_alpha()
        surf = self._scale_if_needed(surf, size)
        return surface_to_texture(surf)

    # ---------- offentlig laster ----------

    def _resolve_textures_base(self) -> Path:
        """
        Finn korrekt assets/textures-katalog robust, uavhengig av hvor vi kjører fra.
        Prøver i rekkefølge:
          - <her>/assets/textures
          - <her>/../assets/textures
          - <her>/../../assets/textures      <-- typisk når koden ligger i src/wolfie3d
          - <cwd>/assets/textures
        """
        here = Path(__file__).resolve().parent
        candidates = [
            here / "assets" / "textures",
            here.parent / "assets" / "textures",
            here.parent.parent / "assets" / "textures",
            Path.cwd() / "assets" / "textures",
        ]
        print("\n[GLRenderer] Prøver å finne assets/textures på disse stedene:")
        for c in candidates:
            print("  -", c)
            if c.exists():
                print("[GLRenderer] FANT:", c)
                return c

        raise FileNotFoundError(
            "Fant ikke assets/textures i noen av kandidatkatalogene over. "
            "Opprett assets/textures på prosjektnivå (samme nivå som src) eller justér stien."
        )

    def load_textures(self) -> None:
        """
        Debug-variant som bruker korrekt prosjekt-rot og feiler høyt hvis filer mangler.
        Forventer: bricks.png, stone.png, wood.png, metal.png i assets/textures/.
        """
        base = self._resolve_textures_base()
        print(f"[GLRenderer] pygame extended image support: {pygame.image.get_extended()}")
        print(f"[GLRenderer] Innhold i {base}: {[p.name for p in base.glob('*')]}")

        files = {
            1: base / "bricks.png",
            2: base / "stone.png",
            3: base / "wood.png",
            4: base / "metal.png",
        }
        missing = [p for p in files.values() if not p.exists()]
        if missing:
            print("[GLRenderer] MANGEL: følgende filer finnes ikke:")
            for m in missing:
                print("  -", m)
            raise FileNotFoundError(
                "Manglende teksturer. Sørg for at filene ligger i assets/textures/")

        def _load(path: Path, size: int = 512) -> int:
            print(f"[GLRenderer] Laster: {path}")
            surf = pygame.image.load(str(path)).convert_alpha()
            if surf.get_width() != size or surf.get_height() != size:
                print(
                    f"[GLRenderer]  - rescale {surf.get_width()}x{surf.get_height()} -> {size}x{size}")
                surf = pygame.transform.smoothscale(surf, (size, size))
            tex_id = surface_to_texture(surf)
            print(f"[GLRenderer]  - OK (GL tex id {tex_id})")
            return tex_id

        self.textures[1] = _load(files[1], 512)
        self.textures[2] = _load(files[2], 512)
        self.textures[3] = _load(files[3], 512)
        self.textures[4] = _load(files[4], 512)

        # Sprite (kule) – behold prosedyre
        self.textures[99] = surface_to_texture(make_bullet_texture())

        # Enemy sprite (ID 200): prøv fil, ellers prosedyral placeholder
        try:
            sprites_dir = self._resolve_textures_base().parent / "sprites"
            enemy_path = sprites_dir / "enemy.png"
            print(f"[GLRenderer] Leter etter enemy sprite i: {enemy_path}")
            if enemy_path.exists():
                self.textures[200] = self._load_texture_file(enemy_path, 512)
                print(f"[GLRenderer] Enemy OK (GL tex id {self.textures[200]})")
            else:
                # fallback – prosedural fiende
                self.textures[200] = surface_to_texture(make_enemy_texture())
                print("[GLRenderer] Enemy: bruker prosedural sprite")
        except Exception as ex:
            print(f"[GLRenderer] Enemy: FEIL ved lasting ({ex}), bruker prosedural")
            self.textures[200] = surface_to_texture(make_enemy_texture())

        # Weapon sprites (ID 300-301): pistol og sau
        try:
            sprites_dir = self._resolve_textures_base().parent / "sprites"
            pistol_path = sprites_dir / "pistol.png"
            sau_path = sprites_dir / "sau.png"
            
            if pistol_path.exists():
                self.textures[300] = self._load_texture_file(pistol_path, 256)
                print(f"[GLRenderer] Pistol OK (GL tex id {self.textures[300]})")
            else:
                print("[GLRenderer] Pistol: fil ikke funnet")
                
            if sau_path.exists():
                self.textures[301] = self._load_texture_file(sau_path, 256)
                print(f"[GLRenderer] Sau OK (GL tex id {self.textures[301]})")
            else:
                print("[GLRenderer] Sau: fil ikke funnet")
        except Exception as ex:
            print(f"[GLRenderer] Våpen: FEIL ved lasting ({ex})")

        print("[GLRenderer] Teksturer lastet.\n")

    # ---------- draw ----------
    def draw_arrays(self, verts: np.ndarray, texture: int, use_tex: bool) -> None:
        if verts.size == 0:
            return
        gl.glUseProgram(self.prog)
        gl.glUniform1i(self.uni_use_tex, 1 if use_tex else 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture if use_tex else self.white_tex)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_DYNAMIC_DRAW)
        count = verts.shape[0]
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, count)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

# ---------- Raycasting + bygg GL-verts ----------
def column_ndc(x: int) -> tuple[float, float]:
    """Returnerer venstre/høyre NDC-X for en 1-px bred skjermkolonne."""
    x_left = (2.0 * x) / WIDTH - 1.0
    x_right = (2.0 * (x + 1)) / WIDTH - 1.0
    return x_left, x_right

def y_ndc(y_pix: int) -> float:
    """Konverter skjerm-Y (0 top) til NDC-Y (1 top, -1 bunn)."""
    return 1.0 - 2.0 * (y_pix / float(HEIGHT))

def dim_for_side(side: int) -> float:
    # dim litt på sidevegger (liknende BLEND_MULT tidligere)
    return 0.78 if side == 1 else 1.0

def cast_and_build_wall_batches() -> dict[int, list[float]]:
    batches: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
    for x in range(WIDTH):
        # Raydir
        camera_x = 2.0 * x / WIDTH - 1.0
        ray_dir_x = dir_x + plane_x * camera_x
        ray_dir_y = dir_y + plane_y * camera_x
        map_x = int(player_x)
        map_y = int(player_y)

        delta_dist_x = abs(1.0 / ray_dir_x) if ray_dir_x != 0 else 1e30
        delta_dist_y = abs(1.0 / ray_dir_y) if ray_dir_y != 0 else 1e30

        if ray_dir_x < 0:
            step_x = -1
            side_dist_x = (player_x - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - player_x) * delta_dist_x
        if ray_dir_y < 0:
            step_y = -1
            side_dist_y = (player_y - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - player_y) * delta_dist_y

        hit = 0
        side = 0
        tex_id = 1
        while hit == 0:
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1
            if not in_map(map_x, map_y):
                hit = 1
                tex_id = 1
                break
            if MAP[map_y][map_x] > 0:
                hit = 1
                tex_id = MAP[map_y][map_x]

        if side == 0:
            perp_wall_dist = (map_x - player_x + (1 - step_x) / 2.0) / (ray_dir_x if ray_dir_x != 0 else 1e-9)
            wall_x = player_y + perp_wall_dist * ray_dir_y
        else:
            perp_wall_dist = (map_y - player_y + (1 - step_y) / 2.0) / (ray_dir_y if ray_dir_y != 0 else 1e-9)
            wall_x = player_x + perp_wall_dist * ray_dir_x

        wall_x -= math.floor(wall_x)
        # u-koordinat (kontinuerlig) + flip for samsvar med klassisk raycaster
        u = wall_x
        if (side == 0 and ray_dir_x > 0) or (side == 1 and ray_dir_y < 0):
            u = 1.0 - u

        # skjermhøyde på vegg
        line_height = int(HEIGHT / (perp_wall_dist + 1e-6))
        draw_start = max(0, -line_height // 2 + HALF_H)
        draw_end = min(HEIGHT - 1, line_height // 2 + HALF_H)

        # NDC koordinater for 1-px bred stripe
        x_left, x_right = column_ndc(x)
        top_ndc = y_ndc(draw_start)
        bot_ndc = y_ndc(draw_end)

        # Farge-dim (samme på hele kolonnen)
        c = dim_for_side(side)
        r = g = b = c

        # Depth som lineær [0..1] (0 = nærmest)
        depth = clamp01(perp_wall_dist / FAR_PLANE)

        # 2 triangler (6 vertikser). Vertex-layout:
        # [x, y, u, v, r, g, b, depth]
        v = [
            # tri 1
            x_left,  top_ndc, u, 0.0, r, g, b, depth,
            x_left,  bot_ndc, u, 1.0, r, g, b, depth,
            x_right, top_ndc, u, 0.0, r, g, b, depth,
            # tri 2
            x_right, top_ndc, u, 0.0, r, g, b, depth,
            x_left,  bot_ndc, u, 1.0, r, g, b, depth,
            x_right, bot_ndc, u, 1.0, r, g, b, depth,
        ]
        batches.setdefault(tex_id, []).extend(v)
    return batches

def build_fullscreen_background() -> np.ndarray:
    """To store quads (himmel/gulv), farget med vertex-color, tegnes uten tekstur."""
    # Himmel (øverst halvdel)
    sky_col = (40/255.0, 60/255.0, 90/255.0)
    floor_col = (35/255.0, 35/255.0, 35/255.0)
    verts: list[float] = []

    # Quad helper
    def add_quad(x0, y0, x1, y1, col):
        r, g, b = col
        depth = 1.0  # lengst bak
        # u,v er 0 (vi bruker hvit 1x1 tekstur)
        verts.extend([
            x0, y0, 0.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,

            x1, y0, 1.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y1, 1.0, 1.0, r, g, b, depth,
        ])

    # Koordinater i NDC
    add_quad(-1.0,  1.0,  1.0,  0.0, sky_col)   # øvre halvdel
    add_quad(-1.0,  0.0,  1.0, -1.0, floor_col) # nedre halvdel
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_sprites_batch(bullets: list[Bullet | SauBullet]) -> np.ndarray:
    """Bygger ett quad per kule/sau i skjermen (billboard), med depth."""
    verts: list[float] = []

    for b in bullets:
        # Transform til kamera-rom
        spr_x = b.x - player_x
        spr_y = b.y - player_y
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
        trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
        trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
        if trans_y <= 0:
            continue  # bak kamera

        sprite_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))

        sprite_h = abs(int(HEIGHT / trans_y))
        sprite_w = sprite_h  # kvadratisk

        # vertikal offset: "stiger"
        v_shift = int((0.5 - b.height_param) * sprite_h)
        draw_start_y = max(0, -sprite_h // 2 + HALF_H + v_shift)
        draw_end_y   = min(HEIGHT - 1, draw_start_y + sprite_h)
        # horisontal
        draw_start_x = -sprite_w // 2 + sprite_screen_x
        draw_end_x   = draw_start_x + sprite_w

        # Klipp utenfor skjerm
        if draw_end_x < 0 or draw_start_x >= WIDTH:
            continue
        draw_start_x = max(0, draw_start_x)
        draw_end_x   = min(WIDTH - 1, draw_end_x)

        # Konverter til NDC
        x0 = (2.0 * draw_start_x) / WIDTH - 1.0
        x1 = (2.0 * (draw_end_x + 1)) / WIDTH - 1.0
        y0 = y_ndc(draw_start_y)
        y1 = y_ndc(draw_end_y)

        # Depth (basert på trans_y)
        depth = clamp01(trans_y / FAR_PLANE)

        # Farge - rød hvis sau er aktiv (kan eksplodere)
        if isinstance(b, SauBullet) and active_sau == b:
            r, g, bcol = 1.0, 0.3, 0.3  # Rød farge for aktiv sau
        else:
            r = g = bcol = 1.0  # ingen ekstra farge-dim
        
        # Teksturkoordinater - flip for sauen
        if isinstance(b, SauBullet):
            # Sau - flip texture coordinates
            u0, v0 = 0.0, 1.0
            u1, v1 = 1.0, 0.0
        else:
            # Kule - normal texture coordinates
            u0, v0 = 0.0, 0.0
            u1, v1 = 1.0, 1.0

        verts.extend([
            x0, y0, u0, v0, r, g, bcol, depth,
            x0, y1, u0, v1, r, g, bcol, depth,
            x1, y0, u1, v0, r, g, bcol, depth,

            x1, y0, u1, v0, r, g, bcol, depth,
            x0, y1, u0, v1, r, g, bcol, depth,
            x1, y1, u1, v1, r, g, bcol, depth,
        ])



    if not verts:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_enemies_batch(enemies: list['Enemy']) -> np.ndarray:
    verts: list[float] = []
    for e in enemies:
        if not e.alive:
            continue
            
        # Håndter dødsanimasjon - synkende effekt
        death_scale = 1.0
        death_offset = 0
        if e.is_dying:
            # Beregn progress av dødsanimasjon (0.0 til 1.0)
            death_progress = min(1.0, e.death_time / 0.3)
            # Synk ned i bakken (øk v_shift)
            death_offset = int(death_progress * 50)  # Synk 50 piksler
            # Krymp fienden (scale ned)
            death_scale = 1.0 - death_progress * 0.5  # Krymp til 50% størrelse
        
        spr_x = e.x - player_x
        spr_y = e.y - player_y
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
        trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
        trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
        if trans_y <= 0:
            continue

        sprite_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))
        sprite_h = abs(int(HEIGHT / trans_y))
        sprite_w = sprite_h  # kvadratisk
        
        # Bruk death_scale for å krympe fienden under dødsanimasjon
        sprite_h = int(sprite_h * death_scale)
        sprite_w = int(sprite_w * death_scale)
        
        v_shift = int((0.5 - e.height_param) * sprite_h) + death_offset

        draw_start_y = max(0, -sprite_h // 2 + HALF_H + v_shift)
        draw_end_y   = min(HEIGHT - 1, draw_start_y + sprite_h)
        draw_start_x = -sprite_w // 2 + sprite_screen_x
        draw_end_x   = draw_start_x + sprite_w
        if draw_end_x < 0 or draw_start_x >= WIDTH:
            continue

        draw_start_x = max(0, draw_start_x)
        draw_end_x   = min(WIDTH - 1, draw_end_x)

        x0 = (2.0 * draw_start_x) / WIDTH - 1.0
        x1 = (2.0 * (draw_end_x + 1)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * (draw_start_y / HEIGHT)
        y1 = 1.0 - 2.0 * (draw_end_y   / HEIGHT)

        depth = clamp01(trans_y / FAR_PLANE)
        
        # Farge - rød/oransje under dødsanimasjon
        if e.is_dying:
            # Eksplosjonsfarge - går fra rød til oransje
            death_progress = min(1.0, e.death_time / 0.3)
            if death_progress < 0.5:
                # Første halvdel: rød
                r, g, b = 1.0, 0.0, 0.0
            else:
                # Andre halvdel: oransje
                r, g, b = 1.0, 0.5, 0.0
        else:
            r = g = b = 1.0

        ENEMY_V_FLIP = True  # sett False hvis den blir riktig uten flip
        if ENEMY_V_FLIP:
            u0, v0, u1, v1 = 0.0, 1.0, 1.0, 0.0
        else:
            u0, v0, u1, v1 = 0.0, 0.0, 1.0, 1.0

        verts.extend([
            x0, y0, u0, v0, r, g, b, depth,
            x0, y1, u0, v1, r, g, b, depth,
            x1, y0, u1, v0, r, g, b, depth,

            x1, y0, u1, v0, r, g, b, depth,
            x0, y1, u0, v1, r, g, b, depth,
            x1, y1, u1, v1, r, g, b, depth,
        ])

    if not verts:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))


def build_crosshair_quads(size_px: int = 8, thickness_px: int = 2) -> np.ndarray:
    """To små rektangler (horisontalt/vertikalt), sentrert i skjermen."""
    verts: list[float] = []

    def rect_ndc(cx, cy, w, h):
        x0 = (2.0 * (cx - w)) / WIDTH - 1.0
        x1 = (2.0 * (cx + w)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * ((cy - h) / HEIGHT)
        y1 = 1.0 - 2.0 * ((cy + h) / HEIGHT)
        return x0, y0, x1, y1

    r = g = b = 1.0
    depth = 0.0  # helt foran

    # horisontal strek
    x0, y0, x1, y1 = rect_ndc(HALF_W, HALF_H, size_px, thickness_px//2)
    verts.extend([
        x0, y0, 0.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y0, 1.0, 0.0, r, g, b, depth,

        x1, y0, 1.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y1, 1.0, 1.0, r, g, b, depth,
    ])

    # vertikal strek
    x0, y0, x1, y1 = rect_ndc(HALF_W, HALF_H, thickness_px//2, size_px)
    verts.extend([
        x0, y0, 0.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y0, 1.0, 0.0, r, g, b, depth,

        x1, y0, 1.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y1, 1.0, 1.0, r, g, b, depth,
    ])

    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_weapon_overlay(firing: bool, recoil_t: float) -> np.ndarray:
    """Våpenoverlay som viser enten pistol eller sau basert på current_weapon."""
    global current_weapon
    
    if current_weapon == WEAPON_PISTOL:
        # Pistol - mindre
        base_w, base_h = 120, 60
    else:  # WEAPON_SAU
        # Sau - større
        base_w, base_h = 180, 120
    
    x = HALF_W - base_w // 2
    y = HEIGHT - base_h - 10
    if firing:
        recoil_amount = 8 if current_weapon == WEAPON_SAU else 6
        y += int(recoil_amount * math.sin(min(1.0, recoil_t) * math.pi))

    x0 = (2.0 * x) / WIDTH - 1.0
    x1 = (2.0 * (x + base_w)) / WIDTH - 1.0
    y0 = 1.0 - 2.0 * (y / HEIGHT)
    y1 = 1.0 - 2.0 * ((y + base_h) / HEIGHT)

    # Hvit farge for tekstur (ingen tinting)
    r, g, b = 1.0, 1.0, 1.0
    depth = 0.0
    
    if current_weapon == WEAPON_SAU:
        # Sau - flip texture coordinates to fix upside down
        verts = [
            x0, y0, 0.0, 1.0, r, g, b, depth,  # Top-left with flipped Y
            x0, y1, 0.0, 0.0, r, g, b, depth,  # Bottom-left with flipped Y
            x1, y0, 1.0, 1.0, r, g, b, depth,  # Top-right with flipped Y

            x1, y0, 1.0, 1.0, r, g, b, depth,  # Top-right with flipped Y
            x0, y1, 0.0, 0.0, r, g, b, depth,  # Bottom-left with flipped Y
            x1, y1, 1.0, 0.0, r, g, b, depth,  # Bottom-right with flipped Y
        ]
    else:
        # Pistol - normal texture coordinates
        verts = [
            x0, y0, 0.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,

            x1, y0, 1.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y1, 1.0, 1.0, r, g, b, depth,
        ]
    
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_weapon_status_display() -> np.ndarray:
    """Viser hvilket våpen som er valgt (liten våpenbilde øverst til høyre)."""
    global current_weapon
    
    if current_weapon == WEAPON_PISTOL:
        # Pistol - mindre
        base_w, base_h = 40, 20
    else:  # WEAPON_SAU
        # Sau - litt større
        base_w, base_h = 50, 30
    
    x = WIDTH - base_w - 10
    y = 10
    
    x0 = (2.0 * x) / WIDTH - 1.0
    x1 = (2.0 * (x + base_w)) / WIDTH - 1.0
    y0 = 1.0 - 2.0 * (y / HEIGHT)
    y1 = 1.0 - 2.0 * ((y + base_h) / HEIGHT)

    # Hvit farge for tekstur
    r, g, b = 1.0, 1.0, 1.0
    depth = 0.0
    
    if current_weapon == WEAPON_SAU:
        # Sau - flip texture coordinates
        verts = [
            x0, y0, 0.0, 1.0, r, g, b, depth,
            x0, y1, 0.0, 0.0, r, g, b, depth,
            x1, y0, 1.0, 1.0, r, g, b, depth,

            x1, y0, 1.0, 1.0, r, g, b, depth,
            x0, y1, 0.0, 0.0, r, g, b, depth,
            x1, y1, 1.0, 0.0, r, g, b, depth,
        ]
    else:
        # Pistol - normal texture coordinates
        verts = [
            x0, y0, 0.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,

            x1, y0, 1.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y1, 1.0, 1.0, r, g, b, depth,
        ]
    
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_hp_display() -> np.ndarray:
    """Viser spiller HP som en rød balk øverst til venstre."""
    global player_hp, player_max_hp
    
    # HP balk
    bar_width = 200
    bar_height = 20
    x = 10
    y = 10
    
    # Bakgrunn (mørk rød)
    bg_x0 = (2.0 * x) / WIDTH - 1.0
    bg_x1 = (2.0 * (x + bar_width)) / WIDTH - 1.0
    bg_y0 = 1.0 - 2.0 * (y / HEIGHT)
    bg_y1 = 1.0 - 2.0 * ((y + bar_height) / HEIGHT)
    
    # HP-fyll (lys rød)
    hp_ratio = max(0.0, player_hp / player_max_hp)
    hp_width = int(bar_width * hp_ratio)
    hp_x0 = (2.0 * x) / WIDTH - 1.0
    hp_x1 = (2.0 * (x + hp_width)) / WIDTH - 1.0
    
    verts = []
    
    # Bakgrunn (mørk rød)
    bg_r, bg_g, bg_b = 0.3, 0.0, 0.0
    verts.extend([
        bg_x0, bg_y0, 0.0, 0.0, bg_r, bg_g, bg_b, 0.0,
        bg_x0, bg_y1, 0.0, 1.0, bg_r, bg_g, bg_b, 0.0,
        bg_x1, bg_y0, 1.0, 0.0, bg_r, bg_g, bg_b, 0.0,
        bg_x1, bg_y0, 1.0, 0.0, bg_r, bg_g, bg_b, 0.0,
        bg_x0, bg_y1, 0.0, 1.0, bg_r, bg_g, bg_b, 0.0,
        bg_x1, bg_y1, 1.0, 1.0, bg_r, bg_g, bg_b, 0.0,
    ])
    
    # HP-fyll (lys rød)
    if hp_ratio > 0.0:
        hp_r, hp_g, hp_b = 1.0, 0.2, 0.2
        verts.extend([
            hp_x0, bg_y0, 0.0, 0.0, hp_r, hp_g, hp_b, 0.0,
            hp_x0, bg_y1, 0.0, 1.0, hp_r, hp_g, hp_b, 0.0,
            hp_x1, bg_y0, 1.0, 0.0, hp_r, hp_g, hp_b, 0.0,
            hp_x1, bg_y0, 1.0, 0.0, hp_r, hp_g, hp_b, 0.0,
            hp_x0, bg_y1, 0.0, 1.0, hp_r, hp_g, hp_b, 0.0,
            hp_x1, bg_y1, 1.0, 1.0, hp_r, hp_g, hp_b, 0.0,
        ])
    
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_score_display() -> np.ndarray:
    """Viser spiller score øverst til høyre."""
    global player_score
    
    # Score boks
    bar_width = 120
    bar_height = 30
    x = WIDTH - bar_width - 10
    y = 50  # Under våpen-status
    
    x0 = (2.0 * x) / WIDTH - 1.0
    x1 = (2.0 * (x + bar_width)) / WIDTH - 1.0
    y0 = 1.0 - 2.0 * (y / HEIGHT)
    y1 = 1.0 - 2.0 * ((y + bar_height) / HEIGHT)
    
    # Gul farge for score
    r, g, b = 1.0, 1.0, 0.0
    depth = 0.0
    
    verts = [
        x0, y0, 0.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y0, 1.0, 0.0, r, g, b, depth,
        x1, y0, 1.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y1, 1.0, 1.0, r, g, b, depth,
    ]
    
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_sau_count_display() -> np.ndarray:
    """Viser antall sauer igjen øverst til høyre."""
    global sau_count, active_sau
    
    # Sau-teller boks
    bar_width = 80
    bar_height = 25
    x = WIDTH - bar_width - 10
    y = 85  # Under score
    
    x0 = (2.0 * x) / WIDTH - 1.0
    x1 = (2.0 * (x + bar_width)) / WIDTH - 1.0
    y0 = 1.0 - 2.0 * (y / HEIGHT)
    y1 = 1.0 - 2.0 * ((y + bar_height) / HEIGHT)
    
    # Farge - rød hvis sau er aktiv, ellers hvit
    if active_sau and active_sau.alive:
        r, g, b = 1.0, 0.3, 0.3  # Rød farge for aktiv sau
    else:
        r, g, b = 1.0, 1.0, 1.0  # Hvit farge for normal
    depth = 0.0
    
    verts = [
        x0, y0, 0.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y0, 1.0, 0.0, r, g, b, depth,
        x1, y0, 1.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y1, 1.0, 1.0, r, g, b, depth,
    ]
    
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def get_player_name_input(renderer) -> str:
    """Henter navn fra spilleren via tastatur."""
    name = ""
    max_length = 15
    

    
    print("Skriv inn ditt navn (maks 15 tegn):")
    print("Trykk ENTER for å bekrefte, ESC for å avbryte")
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return name
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and name.strip():
                    print(f"Navn bekreftet: {name}")
                    return name.strip()
                elif event.key == pygame.K_BACKSPACE:
                    name = name[:-1]
                    print(f"Navn: {name}_")
                elif event.key == pygame.K_ESCAPE:
                    print("Avbrutt")
                    return name
                elif len(name) < max_length:
                    # Kun tillat bokstaver, tall og mellomrom
                    if event.unicode.isalnum() or event.unicode == ' ':
                        name += event.unicode
                        print(f"Navn: {name}_")
        
        # Render input screen
        gl.glViewport(0, 0, WIDTH, HEIGHT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Input boks
        input_width = 400
        input_height = 50
        x = (WIDTH - input_width) // 2
        y = HEIGHT // 2
        
        x0 = (2.0 * x) / WIDTH - 1.0
        x1 = (2.0 * (x + input_width)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * (y / HEIGHT)
        y1 = 1.0 - 2.0 * ((y + input_height) / HEIGHT)
        
        # Hvit farge for input boks
        r, g, b = 1.0, 1.0, 1.0
        depth = 0.0
        
        input_verts = [
            x0, y0, 0.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y1, 1.0, 1.0, r, g, b, depth,
        ]
        
        input_array = np.asarray(input_verts, dtype=np.float32).reshape((-1, 8))
        renderer.draw_arrays(input_array, renderer.white_tex, use_tex=False)
        
        # Render tekst via OpenGL
        draw_text_px(renderer, "Skriv inn ditt navn:", WIDTH//2 - 180, HEIGHT//2 - 50, size=36, color=(255, 255, 255))
        
        # Render navn med cursor
        if len(name) == 0:
            name_display = "_"
        else:
            name_display = name + "_"
        draw_text_px(renderer, name_display, WIDTH//2 - 100, HEIGHT//2, size=36, color=(0, 0, 0))
        
        # Render instruksjoner
        draw_text_px(renderer, "ENTER = bekreft, ESC = avbryt", WIDTH//2 - 230, HEIGHT//2 + 50, size=24, color=(200, 200, 200))
        
        pygame.display.flip()
        
        pygame.time.wait(16)

def build_damage_flash() -> np.ndarray:
    """Bygger rød flash-effekt når spilleren tar skade."""
    global player_damage_flash_time
    
    if player_damage_flash_time <= 0.0:
        return np.zeros((0, 8), dtype=np.float32)
    
    # Flash-effekt som dekker hele skjermen
    # Intensitet avtar over tid
    flash_intensity = min(0.7, player_damage_flash_time / 0.15)  # 0.15 sekunder flash, maks 70% intensitet
    
    # Rød farge med alpha-effekt
    r, g, b = flash_intensity, 0.0, 0.0
    depth = 0.0  # Helt foran
    
    # Fullskjerm quad
    verts = [
        -1.0, -1.0, 0.0, 0.0, r, g, b, depth,  # Bottom-left
        -1.0,  1.0, 0.0, 1.0, r, g, b, depth,  # Top-left
         1.0, -1.0, 1.0, 0.0, r, g, b, depth,  # Bottom-right
         1.0, -1.0, 1.0, 0.0, r, g, b, depth,  # Bottom-right
        -1.0,  1.0, 0.0, 1.0, r, g, b, depth,  # Top-left
         1.0,  1.0, 1.0, 1.0, r, g, b, depth,  # Top-right
    ]
    
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_enemy_spawn_countdown(enemies_spawned: bool, enemy_spawn_timer: float, enemy_spawn_delay: float) -> np.ndarray:
    """Viser nedtelling til fiendene spawner (kun hvis de ikke har spawnet ennå)."""
    
    if enemies_spawned:
        return np.zeros((0, 8), dtype=np.float32)
    
    # Countdown boks
    bar_width = 200
    bar_height = 30
    x = (WIDTH - bar_width) // 2  # Sentrert
    y = HEIGHT - 100  # Nær bunnen
    
    # Beregn progress bar basert på tid
    progress = min(1.0, enemy_spawn_timer / enemy_spawn_delay)
    progress_width = int(bar_width * progress)
    
    x0 = (2.0 * x) / WIDTH - 1.0
    x1 = (2.0 * (x + bar_width)) / WIDTH - 1.0
    y0 = 1.0 - 2.0 * (y / HEIGHT)
    y1 = 1.0 - 2.0 * ((y + bar_height) / HEIGHT)
    
    # Bakgrunn (mørk oransje)
    bg_r, bg_g, bg_b = 0.3, 0.15, 0.0
    depth = 0.0
    
    verts = []
    
    # Bakgrunn
    verts.extend([
        x0, y0, 0.0, 0.0, bg_r, bg_g, bg_b, depth,
        x0, y1, 0.0, 1.0, bg_r, bg_g, bg_b, depth,
        x1, y0, 1.0, 0.0, bg_r, bg_g, bg_b, depth,
        x1, y0, 1.0, 0.0, bg_r, bg_g, bg_b, depth,
        x0, y1, 0.0, 1.0, bg_r, bg_g, bg_b, depth,
        x1, y1, 1.0, 1.0, bg_r, bg_g, bg_b, depth,
    ])
    
    # Progress bar (lys oransje)
    if progress > 0.0:
        progress_x1 = (2.0 * (x + progress_width)) / WIDTH - 1.0
        progress_r, progress_g, progress_b = 1.0, 0.5, 0.0
        
        verts.extend([
            x0, y0, 0.0, 0.0, progress_r, progress_g, progress_b, depth,
            x0, y1, 0.0, 1.0, progress_r, progress_g, progress_b, depth,
            progress_x1, y0, 1.0, 0.0, progress_r, progress_g, progress_b, depth,
            progress_x1, y0, 1.0, 0.0, progress_r, progress_g, progress_b, depth,
            x0, y1, 0.0, 1.0, progress_r, progress_g, progress_b, depth,
            progress_x1, y1, 1.0, 1.0, progress_r, progress_g, progress_b, depth,
        ])
    
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def show_highscores_screen(renderer, final_score: int) -> None:
    """Viser highscores og lar spilleren skrive navn hvis det er en highscore."""
    highscores = load_highscores()
    

    
    # Sjekk om dette er en highscore
    if is_highscore(final_score):
        print(f"🎉 NY HIGHSCORE! Score: {final_score}")
        
        # Hent navn fra spilleren
        player_name = get_player_name_input(renderer)
        if not player_name.strip():
            player_name = "Anonymous"  # Default navn hvis ingen input
        add_highscore(player_name, final_score)
        print(f"Highscore lagret for {player_name}!")
    
    # Vis highscores
    print("\n=== HIGHSCORES ===")
    for i, hs in enumerate(highscores[:10], 1):
        print(f"{i:2d}. {hs['name']:<15} {hs['score']:>6}")
    
    print("\nTrykk ESC for å avslutte...")
    
    # Vis highscores skjerm i 5 sekunder
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < 5000:  # 5 sekunder
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
        
        # Render highscores overlay
        gl.glViewport(0, 0, WIDTH, HEIGHT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Highscores boks
        hs_width = 500
        hs_height = 400
        x = (WIDTH - hs_width) // 2
        y = (HEIGHT - hs_height) // 2
        
        x0 = (2.0 * x) / WIDTH - 1.0
        x1 = (2.0 * (x + hs_width)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * (y / HEIGHT)
        y1 = 1.0 - 2.0 * ((y + hs_height) / HEIGHT)
        
        # Gul farge for highscores boks
        r, g, b = 1.0, 1.0, 0.0
        depth = 0.0
        
        hs_verts = [
            x0, y0, 0.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y1, 1.0, 1.0, r, g, b, depth,
        ]
        
        hs_array = np.asarray(hs_verts, dtype=np.float32).reshape((-1, 8))
        renderer.draw_arrays(hs_array, renderer.white_tex, use_tex=False)
        
        # Render tekst via OpenGL
        draw_text_px(renderer, "HIGHSCORES", WIDTH//2 - 100, y + 30, size=48, color=(0, 0, 0))
        
        # Render highscores
        y_offset = y + 80
        for i, hs in enumerate(highscores[:10], 1):
            score_text = f"{i:2d}. {hs['name']:<15} {hs['score']:>6}"
            draw_text_px(renderer, score_text, x + 20, y_offset, size=24, color=(0, 0, 0))
            y_offset += 25
        
        # Render instruksjoner
        draw_text_px(renderer, "Trykk ESC for å avslutte", WIDTH//2 - 120, y + hs_height - 30, size=24, color=(0, 0, 0))
        
        pygame.display.flip()
        
        pygame.time.wait(16)

def show_game_over_screen(renderer, final_score: int) -> None:
    """Viser Game Over skjerm med final score."""
    print("=== GAME OVER ===")
    print(f"Final Score: {final_score}")
    print("Trykk ESC for å avslutte...")
    

    
    # Vis Game Over skjerm i 3 sekunder
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < 3000:  # 3 sekunder
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
        
        # Render Game Over overlay
        gl.glViewport(0, 0, WIDTH, HEIGHT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)  # Svart bakgrunn
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Game Over tekst (enkel rød balk)
        go_width = 400
        go_height = 100
        x = (WIDTH - go_width) // 2
        y = (HEIGHT - go_height) // 2
        
        x0 = (2.0 * x) / WIDTH - 1.0
        x1 = (2.0 * (x + go_width)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * (y / HEIGHT)
        y1 = 1.0 - 2.0 * ((y + go_height) / HEIGHT)
        
        # Rød farge for Game Over
        r, g, b = 1.0, 0.0, 0.0
        depth = 0.0
        
        go_verts = [
            x0, y0, 0.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y1, 1.0, 1.0, r, g, b, depth,
        ]
        
        go_array = np.asarray(go_verts, dtype=np.float32).reshape((-1, 8))
        renderer.draw_arrays(go_array, renderer.white_tex, use_tex=False)
        
        # Render tekst via OpenGL
        draw_text_px(renderer, "GAME OVER", WIDTH//2 - 200, HEIGHT//2, size=72, color=(255, 255, 255))
        draw_text_px(renderer, f"Final Score: {final_score}", WIDTH//2 - 150, HEIGHT//2 + 60, size=36, color=(255, 255, 255))
        
        pygame.display.flip()
        
        pygame.time.wait(16)  # ~60 FPS

def build_minimap_quads() -> np.ndarray:
    """Liten GL-basert minimap øverst til venstre."""
    scale = 6
    mm_w = MAP_W * scale
    mm_h = MAP_H * scale
    pad = 10
    verts: list[float] = []

    def add_quad_px(x_px, y_px, w_px, h_px, col, depth):
        r, g, b = col
        x0 = (2.0 * x_px) / WIDTH - 1.0
        x1 = (2.0 * (x_px + w_px)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * (y_px / HEIGHT)
        y1 = 1.0 - 2.0 * ((y_px + h_px) / HEIGHT)
        verts.extend([
            x0, y0, 0.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y1, 1.0, 1.0, r, g, b, depth,
        ])

    # Bakgrunn
    add_quad_px(pad-2, pad-2, mm_w+4, mm_h+4, (0.1, 0.1, 0.1), 0.0)

    # Celler
    for y in range(MAP_H):
        for x in range(MAP_W):
            if MAP[y][x] > 0:
                col = (0.86, 0.86, 0.86)
                add_quad_px(pad + x*scale, pad + y*scale, scale-1, scale-1, col, 0.0)

    # Spiller
    px = int(player_x * scale)
    py = int(player_y * scale)
    add_quad_px(pad + px - 2, pad + py - 2, 4, 4, (1.0, 0.3, 0.3), 0.0)

    # Retningsstrek (en liten rektangulær "linje")
    fx = int(px + dir_x * 8)
    fy = int(py + dir_y * 8)
    # tegn som tynn boks mellom (px,py) og (fx,fy)
    # for enkelhet: bare en liten boks på enden
    add_quad_px(pad + fx - 1, pad + fy - 1, 2, 2, (1.0, 0.3, 0.3), 0.0)

    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_explosion_particles(enemies: list['Enemy']) -> np.ndarray:
    """Bygger eksplosjonspartikler for døende fiender."""
    verts: list[float] = []
    
    for e in enemies:
        if not e.is_dying or e.death_time > 0.3:
            continue
            
        # Beregn progress av dødsanimasjon
        death_progress = min(1.0, e.death_time / 0.3)
        
        # Transform fiende til skjermkoordinater
        spr_x = e.x - player_x
        spr_y = e.y - player_y
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
        trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
        trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
        if trans_y <= 0:
            continue

        sprite_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))
        sprite_h = abs(int(HEIGHT / trans_y))
        sprite_w = sprite_h
        
        # Beregn fiendens Y-posisjon på skjermen
        v_shift = int((0.5 - e.height_param) * sprite_h)
        sprite_screen_y = HALF_H + v_shift
        
        # Partikkel-posisjon (rundt fienden)
        particle_count = 12  # Flere partikler for mer effekt
        for i in range(particle_count):
            # Beregn partikkel-posisjon i sirkel rundt fienden
            angle = (i / particle_count) * 2 * math.pi
            radius = 15 + death_progress * 40  # Partiklene sprer seg mer utover
            px = sprite_screen_x + int(radius * math.cos(angle))
            py = sprite_screen_y + int(radius * math.sin(angle))
            
            # Partikkel-størrelse (krymper over tid)
            particle_size = max(3, int(10 * (1.0 - death_progress)))
            
            # Klipp partikkel hvis den går utenfor skjerm
            if px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT:
                continue
                
            # Konverter til NDC
            x0 = (2.0 * (px - particle_size)) / WIDTH - 1.0
            x1 = (2.0 * (px + particle_size)) / WIDTH - 1.0
            y0 = 1.0 - 2.0 * ((py - particle_size) / HEIGHT)
            y1 = 1.0 - 2.0 * ((py + particle_size) / HEIGHT)
            
            # Depth (samme som fienden)
            depth = clamp01(trans_y / FAR_PLANE)
            
            # Partikkelfarge - går fra hvit til gul til rød
            if death_progress < 0.3:
                # Første tredjedel: hvit
                r, g, b = 1.0, 1.0, 1.0
            elif death_progress < 0.7:
                # Andre tredjedel: gul
                r, g, b = 1.0, 1.0, 0.0
            else:
                # Siste tredjedel: rød
                r, g, b = 1.0, 0.0, 0.0
            
            verts.extend([
                x0, y0, 0.0, 0.0, r, g, b, depth,
                x0, y1, 0.0, 1.0, r, g, b, depth,
                x1, y0, 1.0, 0.0, r, g, b, depth,
                x1, y0, 1.0, 0.0, r, g, b, depth,
                x0, y1, 0.0, 1.0, r, g, b, depth,
                x1, y1, 1.0, 1.0, r, g, b, depth,
            ])
    
    if not verts:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_sau_explosion_effect() -> np.ndarray:
    """Bygger eksplosjons-effekt for sau-eksplosjon."""
    global explosion_time, explosion_x, explosion_y
    
    if explosion_time <= 0.0:
        return np.zeros((0, 8), dtype=np.float32)
    
    verts: list[float] = []
    
    # Beregn progress av eksplosjonen (1.0 til 0.0)
    explosion_progress = explosion_time / 0.3
    
    # Transform eksplosjon til skjermkoordinater
    spr_x = explosion_x - player_x
    spr_y = explosion_y - player_y
    inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
    trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
    trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
    if trans_y <= 0:
        return np.zeros((0, 8), dtype=np.float32)

    explosion_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))
    explosion_screen_y = HALF_H
    
    # Eksplosjons-størrelse (vokser over tid)
    explosion_size = int(50 * (1.0 - explosion_progress))
    
    # Klipp eksplosjon hvis den går utenfor skjerm
    if (explosion_screen_x - explosion_size < 0 or 
        explosion_screen_x + explosion_size >= WIDTH or
        explosion_screen_y - explosion_size < 0 or 
        explosion_screen_y + explosion_size >= HEIGHT):
        return np.zeros((0, 8), dtype=np.float32)
    
    # Konverter til NDC
    x0 = (2.0 * (explosion_screen_x - explosion_size)) / WIDTH - 1.0
    x1 = (2.0 * (explosion_screen_x + explosion_size)) / WIDTH - 1.0
    y0 = 1.0 - 2.0 * ((explosion_screen_y - explosion_size) / HEIGHT)
    y1 = 1.0 - 2.0 * ((explosion_screen_y + explosion_size) / HEIGHT)
    
    # Depth (samme som eksplosjonen)
    depth = clamp01(trans_y / FAR_PLANE)
    
    # Eksplosjonsfarge - går fra hvit til gul til oransje
    if explosion_progress > 0.7:
        # Første tredjedel: hvit
        r, g, b = 1.0, 1.0, 1.0
    elif explosion_progress > 0.3:
        # Andre tredjedel: gul
        r, g, b = 1.0, 1.0, 0.0
    else:
        # Siste tredjedel: oransje
        r, g, b = 1.0, 0.5, 0.0
    
    verts.extend([
        x0, y0, 0.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y0, 1.0, 0.0, r, g, b, depth,
        x1, y0, 1.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y1, 1.0, 1.0, r, g, b, depth,
    ])
    
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_enemy_hp_bars(enemies: list['Enemy']) -> np.ndarray:
    """
    Bygger HP-balker som vises over fiendene.
    
    Features:
    - HP-balker som følger fiendene på skjermen
    - Fargekoding: rød (lav HP) -> gul -> grønn (høy HP)
    - Responsiv størrelse basert på fiendens avstand
    - Korrekt depth-testing for 3D-posisjonering
    """
    verts: list[float] = []
    
    for e in enemies:
        if not e.alive:
            continue
            
        # Transform til kamera-rom (samme som for fiendene)
        spr_x = e.x - player_x
        spr_y = e.y - player_y
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
        trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
        trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
        if trans_y <= 0:
            continue

        sprite_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))
        sprite_h = abs(int(HEIGHT / trans_y))
        sprite_w = sprite_h  # kvadratisk
        v_shift = int((0.5 - e.height_param) * sprite_h)

        draw_start_y = max(0, -sprite_h // 2 + HALF_H + v_shift)
        draw_end_y   = min(HEIGHT - 1, draw_start_y + sprite_h)
        draw_start_x = -sprite_w // 2 + sprite_screen_x
        draw_end_x   = draw_start_x + sprite_w
        if draw_end_x < 0 or draw_start_x >= WIDTH:
            continue

        draw_start_x = max(0, draw_start_x)
        draw_end_x   = min(WIDTH - 1, draw_end_x)

        # HP-balk posisjon (over fienden)
        hp_bar_width = sprite_w * 0.8  # 80% av fiendens bredde
        hp_bar_height = 8  # Fast høyde på 8 piksler
        hp_bar_x = sprite_screen_x - hp_bar_width // 2
        hp_bar_y = draw_start_y - hp_bar_height - 5  # 5 piksler over fienden
        
        # Klipp HP-balk hvis den går utenfor skjerm
        if hp_bar_y < 0:
            hp_bar_y = 0
        if hp_bar_y + hp_bar_height > HEIGHT:
            continue
            
        # Konverter til NDC
        x0 = (2.0 * hp_bar_x) / WIDTH - 1.0
        x1 = (2.0 * (hp_bar_x + hp_bar_width)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * (hp_bar_y / HEIGHT)
        y1 = 1.0 - 2.0 * ((hp_bar_y + hp_bar_height) / HEIGHT)

        # Depth (samme som fienden, men litt foran)
        depth = max(0.0, clamp01(trans_y / FAR_PLANE) - 0.01)
        
        # HP-ratio for fyll
        hp_ratio = max(0.0, e.hp / e.max_hp)
        
        # Bakgrunn (mørk rød)
        bg_r, bg_g, bg_b = 0.3, 0.0, 0.0
        verts.extend([
            x0, y0, 0.0, 0.0, bg_r, bg_g, bg_b, depth,
            x0, y1, 0.0, 1.0, bg_r, bg_g, bg_b, depth,
            x1, y0, 1.0, 0.0, bg_r, bg_g, bg_b, depth,
            x1, y0, 1.0, 0.0, bg_r, bg_g, bg_b, depth,
            x0, y1, 0.0, 1.0, bg_r, bg_g, bg_b, depth,
            x1, y1, 1.0, 1.0, bg_r, bg_g, bg_b, depth,
        ])
        
        # HP-fyll (lys rød til grønn basert på HP)
        if hp_ratio > 0.0:
            # Farge går fra rød (lav HP) til grønn (høy HP)
            if hp_ratio > 0.5:
                # Over 50% HP: grønn til gul
                g_ratio = (hp_ratio - 0.5) * 2.0  # 0.0 til 1.0
                hp_r, hp_g, hp_b = 1.0 - g_ratio, 1.0, 0.0
            else:
                # Under 50% HP: rød til gul
                r_ratio = hp_ratio * 2.0  # 0.0 til 1.0
                hp_r, hp_g, hp_b = 1.0, r_ratio, 0.0
            
            # HP-fyll bredde
            hp_fill_width = int(hp_bar_width * hp_ratio)
            hp_fill_x1 = hp_bar_x + hp_fill_width
            
            # Konverter fyll til NDC
            fill_x1 = (2.0 * hp_fill_x1) / WIDTH - 1.0
            
            verts.extend([
                x0, y0, 0.0, 0.0, hp_r, hp_g, hp_b, depth,
                x0, y1, 0.0, 1.0, hp_r, hp_g, hp_b, depth,
                fill_x1, y0, 1.0, 0.0, hp_r, hp_g, hp_b, depth,
                fill_x1, y0, 1.0, 0.0, hp_r, hp_g, hp_b, depth,
                x0, y1, 0.0, 1.0, hp_r, hp_g, hp_b, depth,
                fill_x1, y1, 1.0, 1.0, hp_r, hp_g, hp_b, depth,
            ])

    if not verts:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

# ---------- Input/fysikk ----------
def try_move_player(nx: float, ny: float) -> tuple[float, float]:
    """Wrapper for try_move that uses current player position."""
    return try_move(nx, ny, player_x, player_y)

# ---------- Main ----------
def main() -> None:
    global current_weapon, player_hp, player_invulnerable_time, player_score, sau_count, active_sau, explosion_time, explosion_x, explosion_y, player_x, player_y, dir_x, dir_y, plane_x, plane_y, player_damage_flash_time
    pygame.init()
    pygame.display.set_caption("Vibe Wolf (OpenGL)")

    # setup to make it work on mac as well...
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

    # Opprett GL-kontekst
    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    gl.glViewport(0, 0, WIDTH, HEIGHT)

    clock = pygame.time.Clock()
    renderer = GLRenderer()

    bullets: list[Bullet] = []
    firing = False
    recoil_t = 0.0

    # Spawn enemies at valid floor positions with minimum distance between them
    enemies: list[Enemy] = []
    enemy_types = ["normal", "normal", "strong", "fast", "normal", "strong", "fast", "normal"]
    existing_positions = [(player_x, player_y)]  # Start with player position to avoid spawning too close
    
    # Enemy spawning delay - don't spawn enemies for the first 5 seconds
    enemy_spawn_delay = 5.0  # seconds
    enemy_spawn_timer = 0.0
    enemies_spawned = False

    # Mus-capture (synlig cursor + crosshair)
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(True)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:  # eller en annen knapp
                    grab = not pygame.event.get_grab()
                    pygame.event.set_grab(grab)
                    pygame.mouse.set_visible(not grab)
                    print("Mouse grab:", grab)
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    # Sjekk først om vi har en aktiv sau som kan eksplodere
                    if active_sau and active_sau.alive:
                        # Eksploder sauen
                        explosion_x = active_sau.x
                        explosion_y = active_sau.y
                        explosion_time = 0.3  # 0.3 sekunder eksplosjons-effekt
                        killed_enemies = active_sau.explode(enemies)
                        player_score += len(killed_enemies) * 100  # +100 poeng per fiende drept av eksplosjon
                        active_sau = None  # Fjern aktiv sau
                        print(f"💥 Eksplosjon! {len(killed_enemies)} fiender drept!")
                        firing = True
                        recoil_t = 0.0
                    else:
                        # Skyt basert på valgt våpen
                        if current_weapon == WEAPON_PISTOL:
                            # Pistol - raskere, svakere kuler
                            bx = player_x + dir_x * 0.4
                            by = player_y + dir_y * 0.4
                            bvx = dir_x * 12.0  # Raskere
                            bvy = dir_y * 12.0
                            bullets.append(Bullet(bx, by, bvx, bvy))
                        else:  # WEAPON_SAU
                            # Sau - send sau som prosjektil (kun hvis det er sauer igjen)
                            if sau_count > 0:
                                bx = player_x + dir_x * 0.4
                                by = player_y + dir_y * 0.4
                                bvx = dir_x * 6.0   # Tregere
                                bvy = dir_y * 6.0
                                new_sau = SauBullet(bx, by, bvx, bvy)
                                bullets.append(new_sau)
                                active_sau = new_sau  # Sett den nye sauen som aktiv
                                sau_count -= 1  # Bruk en sau
                                print(f"Sau skutt! {sau_count} sauer igjen")
                            else:
                                print("Ingen sauer igjen!")
                        firing = True
                        recoil_t = 0.0
                if event.key == pygame.K_1:
                    current_weapon = WEAPON_PISTOL
                    print("Våpen 1 (Pistol) valgt")
                if event.key == pygame.K_2:
                    current_weapon = WEAPON_SAU
                    print("Våpen 2 (Sau) valgt")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Sjekk først om vi har en aktiv sau som kan eksplodere
                if active_sau and active_sau.alive:
                    # Eksploder sauen
                    explosion_x = active_sau.x
                    explosion_y = active_sau.y
                    explosion_time = 0.3  # 0.3 sekunder eksplosjons-effekt
                    killed_enemies = active_sau.explode(enemies)
                    player_score += len(killed_enemies) * 100  # +100 poeng per fiende drept av eksplosjon
                    active_sau = None  # Fjern aktiv sau
                    print(f"💥 Eksplosjon! {len(killed_enemies)} fiender drept!")
                    firing = True
                    recoil_t = 0.0
                else:
                    # Skyt basert på valgt våpen (samme som SPACE)
                    if current_weapon == WEAPON_PISTOL:
                        # Pistol - raskere, svakere kuler
                        bx = player_x + dir_x * 0.4
                        by = player_y + dir_y * 0.4
                        bvx = dir_x * 12.0  # Raskere
                        bvy = dir_y * 12.0
                        bullets.append(Bullet(bx, by, bvx, bvy))
                    else:  # WEAPON_SAU
                        # Sau - send sau som prosjektil (kun hvis det er sauer igjen)
                        if sau_count > 0:
                            bx = player_x + dir_x * 0.4
                            by = player_y + dir_y * 0.4
                            bvx = dir_x * 6.0   # Tregere
                            bvy = dir_y * 6.0
                            new_sau = SauBullet(bx, by, bvx, bvy)
                            bullets.append(new_sau)
                            active_sau = new_sau  # Sett den nye sauen som aktiv
                            sau_count -= 1  # Bruk en sau
                            print(f"Sau skutt! {sau_count} sauer igjen")
                        else:
                            print("Ingen sauer igjen!")
                    firing = True
                    recoil_t = 0.0

        # Handle input using the new module
        new_player_x, new_player_y, new_dir_x, new_dir_y, new_plane_x, new_plane_y = handle_input(
            dt, player_x, player_y, dir_x, dir_y, plane_x, plane_y, try_move_player
        )
        player_x, player_y = new_player_x, new_player_y
        dir_x, dir_y = new_dir_x, new_dir_y
        plane_x, plane_y = new_plane_x, new_plane_y

        # Enemy spawning logic with delay
        if not enemies_spawned:
            enemy_spawn_timer += dt
            if enemy_spawn_timer >= enemy_spawn_delay:
                # Spawn all enemies after the delay
                for enemy_type in enemy_types:
                    x, y = find_valid_spawn_position(existing_positions, min_distance=3.0)
                    enemies.append(Enemy(x, y, enemy_type))
                    existing_positions.append((x, y))
                enemies_spawned = True
                print(f"Fiender spawnet etter {enemy_spawn_delay} sekunder!")

        # Oppdater bullets
        for b in bullets:
            b.update(dt)
            if not b.alive:
                continue
            for e in enemies:
                if not e.alive:
                    continue
                dx = e.x - b.x
                dy = e.y - b.y
                if dx * dx + dy * dy <= (e.radius * e.radius):
                    if isinstance(b, SauBullet):
                        # Sau bouncer bort fra fienden
                        b.bounce(e.x, e.y)
                        active_sau = b  # Sett denne sauen som aktiv for eksplosjon
                        break
                    else:
                        # Vanlig kule gjør skade
                        damage = 1
                        e.hp -= damage
                        b.alive = False  # kula forbrukes
                        if e.hp <= 0:
                            e.is_dying = True  # Start dødsanimasjon
                            e.death_time = 0.0
                            player_score += 100  # +100 poeng per fiende drept
                            print(f"Fiende drept! HP: {e.hp}/{e.max_hp} | Score: {player_score}")
                        else:
                            print(f"Fiende skadet! HP: {e.hp}/{e.max_hp}")
                        break
        bullets = [b for b in bullets if b.alive]
        
        # Fjern aktiv sau hvis den ikke lenger er alive
        if active_sau and not active_sau.alive:
            active_sau = None

        # Oppdater fiender og sjekk spiller-skade
        for e in enemies:
            e.update(dt, player_x, player_y)
            
            # Sjekk om fiende er nær nok til å skade spilleren
            if e.alive and not e.is_dying:  # Ikke skade hvis fienden er døende
                dx = player_x - e.x
                dy = player_y - e.y
                dist = math.hypot(dx, dy)
                
                # Fiende skader spilleren hvis den er nær, spilleren ikke er invulnerable, og 300ms har gått
                current_time = pygame.time.get_ticks() / 1000.0
                if (dist <= 0.8 and player_invulnerable_time <= 0.0 and 
                    current_time - e.last_damage_time >= 0.3):  # 300ms forsinkelse
                    player_hp -= 10  # 10 skade per frame når fiende er nær
                    player_invulnerable_time = 0.5  # 0.5 sekunder invulnerability
                    player_damage_flash_time = 0.15  # 0.15 sekunder rød flash
                    e.last_damage_time = current_time  # Oppdater siste skade-tid
                    print(f"Spiller tar skade! HP: {player_hp}/{player_max_hp}")
                    
                    # Sjekk om spilleren dør
                    if player_hp <= 0:
                        print(f"GAME OVER - Spilleren døde! Final Score: {player_score}")
                        show_highscores_screen(renderer, player_score)
                        running = False
        
        # Oppdater invulnerability timer
        if player_invulnerable_time > 0.0:
            player_invulnerable_time -= dt
            
        # Oppdater damage flash timer
        if player_damage_flash_time > 0.0:
            player_damage_flash_time -= dt
            
        # Oppdater eksplosjons timer
        if explosion_time > 0.0:
            explosion_time -= dt

        # Sjekk om spilleren har vunnet (drept alle fiender)
        if enemies_spawned and all(not e.alive for e in enemies):
            print(f"🎉 VICTORY! Alle fiender drept! Final Score: {player_score}")
            show_highscores_screen(renderer, player_score)
            running = False

        # ---------- Render ----------
        gl.glViewport(0, 0, WIDTH, HEIGHT)
        gl.glClearColor(0.05, 0.07, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Bakgrunn (himmel/gulv)
        bg = build_fullscreen_background()
        renderer.draw_arrays(bg, renderer.white_tex, use_tex=False)

        # Vegger (batch pr. tex_id)
        batches_lists = cast_and_build_wall_batches()
        for tid, verts_list in batches_lists.items():
            if tid not in renderer.textures:
                continue
            if not verts_list:
                continue
            arr = np.asarray(verts_list, dtype=np.float32).reshape((-1, 8))
            renderer.draw_arrays(arr, renderer.textures[tid], use_tex=True)

        # Sprites (kuler og sauer)
        # Separer kuler og sauer
        bullet_list = [b for b in bullets if isinstance(b, Bullet)]
        sau_list = [b for b in bullets if isinstance(b, SauBullet)]
        
        # Render kuler
        if bullet_list:
            bullet_spr = build_sprites_batch(bullet_list)
            if bullet_spr.size:
                renderer.draw_arrays(bullet_spr, renderer.textures[99], use_tex=True)
        
        # Render sauer
        if sau_list:
            sau_spr = build_sprites_batch(sau_list)
            if sau_spr.size:
                renderer.draw_arrays(sau_spr, renderer.textures[301], use_tex=True)

        # Enemies (billboards)
        enemies_batch = build_enemies_batch(enemies)
        if enemies_batch.size:
            renderer.draw_arrays(enemies_batch, renderer.textures[200], use_tex=True)

        # Explosion particles (må tegnes etter fiendene)
        explosion_particles = build_explosion_particles(enemies)
        if explosion_particles.size:
            renderer.draw_arrays(explosion_particles, renderer.white_tex, use_tex=False)
            
        # Sau eksplosjons-effekt
        sau_explosion = build_sau_explosion_effect()
        if sau_explosion.size:
            renderer.draw_arrays(sau_explosion, renderer.white_tex, use_tex=False)

        # Enemy HP bars (må tegnes etter fiendene, men før UI-elementer)
        enemy_hp_bars = build_enemy_hp_bars(enemies)
        if enemy_hp_bars.size:
            renderer.draw_arrays(enemy_hp_bars, renderer.white_tex, use_tex=False)

        # Crosshair
        cross = build_crosshair_quads(8, 2)
        renderer.draw_arrays(cross, renderer.white_tex, use_tex=False)

        # Weapon overlay
        if firing:
            recoil_t += dt
            if recoil_t > 0.15:
                firing = False
        overlay = build_weapon_overlay(firing, recoil_t)
        
        # Velg riktig våpenbilde basert på current_weapon
        if current_weapon == WEAPON_PISTOL and 300 in renderer.textures:
            weapon_tex = renderer.textures[300]  # Pistol
        elif current_weapon == WEAPON_SAU and 301 in renderer.textures:
            weapon_tex = renderer.textures[301]  # Sau
        else:
            weapon_tex = renderer.white_tex  # Fallback
        
        renderer.draw_arrays(overlay, weapon_tex, use_tex=True)

        # Minimap
        mm = build_minimap_quads()
        renderer.draw_arrays(mm, renderer.white_tex, use_tex=False)

        # HP display
        hp_display = build_hp_display()
        renderer.draw_arrays(hp_display, renderer.white_tex, use_tex=False)

        # Weapon status display
        status_display = build_weapon_status_display()
        
        # Velg riktig våpenbilde for statusindikator
        if current_weapon == WEAPON_PISTOL and 300 in renderer.textures:
            status_tex = renderer.textures[300]  # Pistol
        elif current_weapon == WEAPON_SAU and 301 in renderer.textures:
            status_tex = renderer.textures[301]  # Sau
        else:
            status_tex = renderer.white_tex  # Fallback
        
        renderer.draw_arrays(status_display, status_tex, use_tex=True)

        # Score display
        score_display = build_score_display()
        renderer.draw_arrays(score_display, renderer.white_tex, use_tex=False)
        
        # Render score text
        draw_text_px(renderer, f"Score: {player_score}", WIDTH - 125, 55, size=20, color=(0, 0, 0))

        # Sau count display
        sau_count_display = build_sau_count_display()
        renderer.draw_arrays(sau_count_display, renderer.white_tex, use_tex=False)
        
        # Render sheep count text
        if active_sau and active_sau.alive:
            draw_text_px(renderer, f"Sau: EKSPLODER!", WIDTH - 85, 90, size=20, color=(255, 0, 0))
        else:
            draw_text_px(renderer, f"Sauer: {sau_count}", WIDTH - 85, 90, size=20, color=(0, 0, 0))

        # Enemy spawn countdown (kun hvis fiendene ikke har spawnet ennå)
        countdown_display = build_enemy_spawn_countdown(enemies_spawned, enemy_spawn_timer, enemy_spawn_delay)
        if countdown_display.size:
            renderer.draw_arrays(countdown_display, renderer.white_tex, use_tex=False)

        # Damage flash effect (må tegnes sist, over alt annet)
        damage_flash = build_damage_flash()
        if damage_flash.size:
            renderer.draw_arrays(damage_flash, renderer.white_tex, use_tex=False)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)

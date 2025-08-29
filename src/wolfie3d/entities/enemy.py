#!/usr/bin/env python3
"""
Enemy class for Wolfie3D game
"""

import math
from ..world.map import is_wall


class Enemy:
    """Enemy entity with different types and behaviors."""
    
    def __init__(self, x: float, y: float, enemy_type: str = "normal") -> None:
        self.x = x
        self.y = y
        self.alive = True
        self.enemy_type = enemy_type
        
        # Forskjellige fiendetyper med ulike egenskaper
        if enemy_type == "strong":
            self.max_hp = 5        # Sterk fiende - 5 HP
            self.speed = 0.8       # Tregere
            self.radius = 0.4      # Større hitbox
            self.height_param = 0.6  # Litt høyere
        elif enemy_type == "fast":
            self.max_hp = 2        # Rask fiende - 2 HP
            self.speed = 2.2       # Raskere
            self.radius = 0.3      # Mindre hitbox
            self.height_param = 0.4  # Litt lavere
        else:  # normal
            self.max_hp = 3        # Vanlig fiende - 3 HP
            self.speed = 1.4       # Normal fart
            self.radius = 0.35     # Normal hitbox
            self.height_param = 0.5  # Normal høyde
            
        self.hp = self.max_hp  # Nåværende HP
        self.animation_time = 0.0  # For animasjon
        self.animation_frame = 0   # Hvilken frame vi er på
        self.death_time = 0.0      # For dødsanimasjon
        self.is_dying = False      # Er fienden i dødsanimasjon?
        self.last_damage_time = 0.0  # Tid da fienden sist skadet spilleren

    def _try_move(self, nx: float, ny: float) -> None:
        # enkel vegg-kollisjon (sirkulær hitbox mot grid)
        # prøv X:
        if not is_wall(int(nx), int(self.y)):
            self.x = nx
        # prøv Y:
        if not is_wall(int(self.x), int(ny)):
            self.y = ny

    def update(self, dt: float, player_x: float, player_y: float) -> None:
        if not self.alive:
            return
        
        # Håndter dødsanimasjon
        if self.is_dying:
            self.death_time += dt
            if self.death_time > 0.3:  # Dødsanimasjon varer 0.3 sekunder (raskere)
                self.alive = False
                return
        
        # Oppdater animasjon (kun hvis ikke døende)
        if not self.is_dying:
            self.animation_time += dt
            if self.animation_time > 0.3:  # Bytt frame hver 0.3 sekunder
                self.animation_frame = (self.animation_frame + 1) % 2  # 2 frames (0 og 1)
                self.animation_time = 0.0
        
        # enkel "chase": gå mot spilleren, stopp om rett foran vegg (kun hvis ikke døende)
        if not self.is_dying:
            dx = player_x - self.x
            dy = player_y - self.y
            dist = math.hypot(dx, dy) + 1e-9
            # ikke gå helt oppå spilleren
            if dist > 0.75:
                ux, uy = dx / dist, dy / dist
                step = self.speed * dt
                self._try_move(self.x + ux * step, self.y + uy * step)

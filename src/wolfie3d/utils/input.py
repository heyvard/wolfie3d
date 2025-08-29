#!/usr/bin/env python3
"""
Input handling for Wolfie3D game
"""

import math
import pygame
from ..utils.constants import MOVE_SPEED, ROT_SPEED, STRAFE_SPEED


def handle_input(dt: float, player_x: float, player_y: float, 
                dir_x: float, dir_y: float, plane_x: float, plane_y: float,
                try_move_func) -> tuple[float, float, float, float, float, float]:
    """
    Handle keyboard input for player movement and rotation.
    
    Returns:
        tuple: (new_player_x, new_player_y, new_dir_x, new_dir_y, new_plane_x, new_plane_y)
    """
    keys = pygame.key.get_pressed()
    
    # Rotation
    rot = 0.0
    if keys[pygame.K_LEFT] or keys[pygame.K_q]:
        rot -= ROT_SPEED * dt
    if keys[pygame.K_RIGHT] or keys[pygame.K_e]:
        rot += ROT_SPEED * dt
    
    new_dir_x, new_dir_y = dir_x, dir_y
    new_plane_x, new_plane_y = plane_x, plane_y
    
    if rot != 0.0:
        cosr, sinr = math.cos(rot), math.sin(rot)
        new_dir_x = dir_x * cosr - dir_y * sinr
        new_dir_y = dir_x * sinr + dir_y * cosr
        new_plane_x = plane_x * cosr - plane_y * sinr
        new_plane_y = plane_x * sinr + plane_y * cosr

    # Forward/backward movement
    forward = 0.0
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        forward += MOVE_SPEED * dt
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        forward -= MOVE_SPEED * dt
    
    new_player_x, new_player_y = player_x, player_y
    if forward != 0.0:
        nx = player_x + new_dir_x * forward
        ny = player_y + new_dir_y * forward
        new_player_x, new_player_y = try_move_func(nx, ny)

    # Strafe movement
    strafe = 0.0
    if keys[pygame.K_a]:
        strafe -= STRAFE_SPEED * dt
    if keys[pygame.K_d]:
        strafe += STRAFE_SPEED * dt
    
    if strafe != 0.0:
        nx = new_player_x + (-new_dir_y) * strafe
        ny = new_player_y + (new_dir_x) * strafe
        new_player_x, new_player_y = try_move_func(nx, ny)

    return new_player_x, new_player_y, new_dir_x, new_dir_y, new_plane_x, new_plane_y

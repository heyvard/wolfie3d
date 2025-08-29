#!/usr/bin/env python3
"""
Map and collision handling for Wolfie3D game
"""

from ..utils.constants import MAP, MAP_W, MAP_H


def in_map(ix: int, iy: int) -> bool:
    """Check if coordinates are within map bounds."""
    return 0 <= ix < MAP_W and 0 <= iy < MAP_H


def is_wall(ix: int, iy: int) -> bool:
    """Check if position contains a wall."""
    return in_map(ix, iy) and MAP[iy][ix] > 0


def try_move(nx: float, ny: float, current_x: float, current_y: float) -> tuple[float, float]:
    """
    Try to move to new position, handling wall collisions.
    
    Args:
        nx: New x position
        ny: New y position
        current_x: Current x position
        current_y: Current y position
        
    Returns:
        tuple: (final_x, final_y) - actual position after collision handling
    """
    if not is_wall(int(nx), int(current_y)):
        x = nx
    else:
        x = current_x
    if not is_wall(int(x), int(ny)):
        y = ny
    else:
        y = current_y
    return x, y


def clamp01(x: float) -> float:
    """Clamp value between 0.0 and 1.0."""
    if x < 0.0: 
        return 0.0
    if x > 1.0: 
        return 1.0
    return x

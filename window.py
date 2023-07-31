import pygame
pygame.init()
import numpy as np
display = pygame.display.set_mode((350, 350))
pygame.display.set_caption("FMCW")

run = True

arr = np.random.rand(500,150)
surf = pygame.surfarray.make_surface(arr)
while run:
    # opóźnienie w grze
    pygame.time.delay(50)
    arr = np.random.rand(500, 150)
    surf = pygame.surfarray.make_surface(arr)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    display.blit(surf, (0, 0))
    pygame.display.update()


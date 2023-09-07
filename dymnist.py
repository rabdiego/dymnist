import pygame as pg
import numpy as np
import tensorflow as tf

pg.init()

SIDE_LENGHT = 560
PIXEL_SIZE = SIDE_LENGHT/28
display = pg.display.set_mode((SIDE_LENGHT, SIDE_LENGHT))
clock = pg.time.Clock()

canvas = np.zeros((28, 28))
rects = [[pg.Rect(i*PIXEL_SIZE, j*PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE) for i in range(28)] for j in range(28)]

font = pg.font.Font('freesansbold.ttf', 36)

model = tf.keras.Sequential([
    tf.keras.layers.RandomZoom(height_factor=0.6, width_factor=0.6, fill_mode='constant'),
    tf.keras.layers.Conv2D(input_shape=(28,28,1),filters=8, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=8, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.GlobalMaxPool2D(),
    tf.keras.layers.Dense(units=120, activation='relu'),
    tf.keras.layers.Dense(units=84, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.build((None, 28, 28, 1))

model.load_weights('models/cp.ckpt')

text = font.render(f'Hi', True, (255,255,255))
text_rect = text.get_rect()
text_rect.center = (500, 100)  

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                label = np.argmax(model.predict(np.array([canvas])), axis=1)
                text = font.render(f'{label[0]}', True, (255,255,255))
            elif event.key == pg.K_SPACE:
                canvas = np.zeros((28, 28))
    
    is_mouse_being_pressed = pg.mouse.get_pressed()
    if is_mouse_being_pressed[0]:
        pos = np.array(pg.mouse.get_pos())
        pos = pos//20
        canvas[pos[1], pos[0]] = 1
    elif is_mouse_being_pressed[2]:
        pos = np.array(pg.mouse.get_pos())
        pos = pos//20
        canvas[pos[1], pos[0]] = 0
    
    for i in range(28):
        for j in range(28):
            pg.draw.rect(display, [canvas[i][j]*255 for _ in range(3)], rects[i][j])
    
    
    display.blit(text, text_rect)
    
    pg.display.update()
    clock.tick(60)

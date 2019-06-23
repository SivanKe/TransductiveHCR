import ctypes,os


ctypes.CDLL("libIDCodbc.so", mode = ctypes.RTLD_GLOBAL)
ctypes.CDLL("libIDCodbc.so", mode = ctypes.RTLD_GLOBAL)
ctypes.CDLL("libIDCodbc.so", mode = ctypes.RTLD_GLOBAL)
_render_text = ctypes.CDLL('bin/create_text_image.so', mode=ctypes.RTLD_GLOBAL)
_render_text.create_text_image.argtypes = (ctypes.c_char_p, ctypes.c_char_p)

FONT_WEIGHTS = ["ultralight", "light", "normal", "bold", "ultrabold", "heavy"]
FONT_STRETCHES = ["ultracondensed", "extracondensed", "condensed", "semicondensed",
  "normal", "semiexpanded", "expanded", "extraexpanded", "ultraexpanded"]

def render_text(text, font, weight, stretch):
    global _render_text
    font_path = os.path.join('../Fonts/', font + '.ttf')
    text = "<span foreground='black' background='white' font_family='{}' font_weight='{}' font_stretch='{}'>".format(
        font, FONT_WEIGHTS[weight], FONT_STRETCHES[stretch]
    ) + text + '</span>'
    result = _render_text.create_text_image(ctypes.c_char_p(text), ctypes.c_char_p(font_path))
    return int(result)

if __name__ == '__main__':
    with open('text_example.txt', 'r') as f:
        text_to_render = f.readlines(f)
        text_to_render = ''.join(text_to_render)
    font = 'Qomolangma-Betsu'
    weight = 0
    stretch = 0
    render_text(text_to_render, font, weight, stretch)
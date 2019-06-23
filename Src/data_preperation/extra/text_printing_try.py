# -*- coding: utf-8 -*-
import cairo
import pango
import pangocairo
import sys

surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 320, 120)
context = cairo.Context(surf)

#draw a background rectangle:
context.rectangle(0,0,320,120)
context.set_source_rgb(1, 1, 1)
context.fill()

#get font families:

font_map = pangocairo.cairo_font_map_get_default()
families = font_map.list_families()

# to see family names:
print([f.get_name() for f in   font_map.list_families()])

#context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)

# Translates context so that desired text upperleft corner is at 0,0
context.translate(50,25)

pangocairo_context = pangocairo.CairoContext(context)
pangocairo_context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)

layout = pangocairo_context.create_layout()
fontname = sys.argv[1] if len(sys.argv) >= 2 else "Sans"
font = pango.FontDescription(fontname + " 25")
layout.set_font_description(font)

layout.set_text(u"Hello World")
context.set_source_rgb(0, 0, 0)
pangocairo_context.update_layout(layout)
pangocairo_context.show_layout(layout)

with open("cairo_text.png", "wb") as image_file:
    surf.write_to_png(image_file)
gcc -c -fPIC -I/usr/include/cairo -I/usr/include/freetype2 -I/usr/include/pango-1.0 -I/usr/include/glib-2.0 `pkg-config --cflags cairo` `pkg-config --cflags freetype2` `pkg-config --cflags gtk+-2.0` `pkg-config --cflags pango` -o bin/render_text_image.o c_code/render_text_image.c -lfontconfig -lcairo

gcc -shared -o bin/render_text_image.so bin/render_text_image.o
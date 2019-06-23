# Makefile for Writing Make Files Example

# *****************************************************
# Variables to control Makefile operation

CXX = gcc
CXXFLAGS = -Wall -g -fPIC -I/usr/include/cairo -I/usr/include/freetype2/freetype -I/usr/include/pango-1.0 -I/usr/include/glib-2.0 \
`pkg-config --cflags cairo` `pkg-config --cflags freetype2` `pkg-config --cflags gtk+-2.0` `pkg-config --cflags pango`
LDFLAGS = `pkg-config --libs cairo` `pkg-config --libs freetype2` `pkg-config --libs gtk+-2.0` `pkg-config --libs pango`
DEPS = render_text_image.h

ODIR=../bin
OBJ = $(ODIR)/main.o $(ODIR)/render_text_image.o
SHOW_FONTS_OBJ = $(ODIR)/main_show_fonts.o
dir_guard=@mkdir -p $(ODIR)

# ****************************************************
# Targets needed to bring the executable up to date

all: main main_show_fonts

main: $(OBJ)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $(ODIR)/$@

main_show_fonts: $(SHOW_FONTS_OBJ)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $(ODIR)/$@


# The main.o target can be written more simply

$(ODIR)/%.o: %.c $(DEPS)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(LDFLAGS)



/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.c
 *
 * Created on February 21, 2018, 10:27 AM
 */

/*
 
  Text layouting and rendering with Pango
  ---------------------------------------
 
  This code snippet shows how to create a cairo surface and 
  render some text into it using Pango. We store the generated
  pixels in a png file.
 
  n.b. this file was created for testing not production.
 
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <cairo.h>
#include <ftbitmap.h>
#include <pango/pangocairo.h>
#include <pango/pangoft2.h>
#include <gtk-2.0/gtk/gtk.h>
 
#define USE_FREETYPE 1
#define USE_RGBA 0
 
int create_text_image(char * out_path, char * text, char * font_path) {
  
  const FcChar8 * file = (const FcChar8 *)font_path;
  FcConfig *fontConfig = FcInitLoadConfigAndFonts();
  // FcBool fontAddStatus = FcConfigAppFontAddFile(FcConfigGetCurrent(), file);
  if (!FcConfigAppFontAddFile(fontConfig, file)){
    exit(EXIT_FAILURE);
  }
  FcConfigBuildFonts( fontConfig );
  FcConfigSetCurrent( fontConfig );


  // cairo_surface_t* surf = NULL;
  // cairo_t* cr = NULL;
  cairo_status_t status;
  PangoContext* context = NULL;
  PangoLayout* layout = NULL;
  // PangoFontDescription* font_desc = NULL;
  PangoFontMap* font_map = NULL;
  // FT_Bitmap bmp = {0};
 
  // int stride = 0;
 
  /* ------------------------------------------------------------ */
  /*               D R A W   I N T O  C A N V A S                 */
  /* ------------------------------------------------------------ */
 
  font_map = pango_cairo_font_map_new();

  if (NULL == font_map) {
    printf("+ error: cannot create the pango font map.\n");
    exit(EXIT_FAILURE);
  }
  
  context = pango_font_map_create_context(font_map);
  if (NULL == context) {
    printf("+ error: cannot create pango font context.\n");
    exit(EXIT_FAILURE);
  }

  FcPattern *p = FcPatternCreate();
  FcObjectSet *os = FcObjectSetBuild(FC_FAMILY,NULL);
  FcFontSet *fs = FcFontList(fontConfig, p, os);
  FcPatternDestroy( p );
  FcObjectSetDestroy( os );
  int i;
  for( i = 0; i < fs->nfont; ++i )
  {
        guchar* fontName = FcNameUnparse( fs->fonts[i] );
        PangoFontDescription* fontDesc = pango_font_description_from_string( (gchar*)fontName );
        pango_font_map_load_font( font_map, context, fontDesc );
        pango_font_description_free( fontDesc );
        g_free(fontName);
  }
 
  /* create layout object. */
  layout = pango_layout_new(context);
  if (NULL == layout) {
    printf("+ error: cannot create the pango layout.\n");
    exit(EXIT_FAILURE);
  }
  pango_layout_set_markup(layout, text, -1);
  /*pango_layout_set_markup(layout, text, -1);*/
  /* create the font description @todo the reference does not tell how/when to free this */
  // char font_string[256];
  /*snprintf(font_string, sizeof font_string, "%s %s %i %s %s", 
          font_family, font_weight, FONT_WEIGHTS[font_weight], FONT_STRETCHES[font_stretch]);*/
  /*pango_font_description_set_weight(font_desc, FONT_WEIGHTS[font_weight]);
  pango_font_description_set_stretch(font_desc, pango.STRETCH_SEMI_CONDENSED);*/
  pango_layout_set_single_paragraph_mode (layout,0);
  pango_layout_set_spacing (layout,0);

  pango_layout_context_changed(layout);
  
  int height;
  int width;
  pango_layout_get_pixel_size(layout, &width, &height);
  height = height + 30 ;
  width = width + 30;
  
  cairo_surface_t *cairo_surface = cairo_image_surface_create (CAIRO_FORMAT_RGB24,
                            width,
                            height);
  
  cairo_t *cairo_context = cairo_create(cairo_surface);
  cairo_move_to(cairo_context, 0, 0);
  
  cairo_set_source_rgb (cairo_context, 1, 1, 1);
  cairo_rectangle(cairo_context, 0, 0, width, height);
  cairo_fill(cairo_context);
  cairo_move_to(cairo_context, 15, 15);
  pango_cairo_update_layout( cairo_context, layout );
  pango_cairo_show_layout( cairo_context, layout ); 

  status = cairo_surface_write_to_png(cairo_surface, out_path);

  cairo_surface_destroy (cairo_surface);

  if (status != CAIRO_STATUS_SUCCESS)
    {
      g_printerr ("Could not save png to '%s'\n", out_path);
      return 1;
    }

  return 0;
}

#include <glib.h>
#include <pango/pangocairo.h>
#include <string.h>
#include <stdlib.h>

#include "render_text_image.h"


int main (int argc, char ** argv)
{
    if (argc != 9){
        printf("Error: must recieve two parameters - <text to print> <output path> <font directory> <font family>"
        "<font weight> <font consetration>\n");
        return -1;
    }
    const char * const FONT_WEIGHTS[] = { "ultralight", "light", "normal", "bold", "ultrabold", "heavy" };
    const char * const FONT_STRETCHES[] = { "ultracondensed", "extracondensed", "condensed", "semicondensed", \
  "normal", "semiexpanded", "expanded", "extraexpanded", "ultraexpanded" };
    const char * const LETTER_SPACING[] = { "0", "1000", "2000" };
    const char * const FONT_SIZE[] = { "10000", "150000", "20000", "25000" };
    const char *font_family = argv[4];

    char *out_path = malloc(sizeof(char) * (strlen(argv[2]) + 1));
    strcpy(out_path, argv[2]);

    const char * font_weight = FONT_WEIGHTS[atoi(argv[5])];
    const char * font_stretch = FONT_STRETCHES[atoi(argv[6])];
    const char * letter_spacing = LETTER_SPACING[atoi(argv[7])];
    const char * font_size = FONT_SIZE[atoi(argv[8])];


    char *murkup_pattern = "<span foreground='black' background='white' font_family='%s' font_size='%s' font_weight='%s' font_stretch='%s' letter_spacing=%s></span>";
    //printf("%s\n",murkup_pattern);
    char *murkup_text = malloc(sizeof(char) * (strlen(murkup_pattern) + strlen(font_weight) + strlen(font_stretch) \
    + strlen(font_family) + strlen(argv[1]) + strlen(letter_spacing) + 10 + strlen(font_size) )) ;
    //printf("%s\n",murkup_pattern);
    sprintf(murkup_text, "<span foreground='black' background='white' font_family='%s' font_size='%s' font_weight='%s' font_stretch='%s' letter_spacing='%s'>%s</span>",
     font_family, font_size, font_weight, font_stretch, letter_spacing,argv[1]);
    create_text_image(out_path, murkup_text, argv[3]);
    //free(out_path);
    //free(murkup_text);
    return 0;
     
}

package Lingua::BO::Phonetics::Padmakara;

use strict;
use utf8;

use Carp qw/croak/;
use base qw/Lingua::BO::Phonetics/;

# some stuff is kept around and re-used when multiple Lingua::BO::Phonetics::Padmakara
# objects are loaded.  most importantly we reuse a single copy of the word list.
our (%preloaded_object, %exceptions, %words, $wylie_object, $lists_loaded);

=head1 NAME

Lingua::BO::Phonetics::Padmakara - Perl extension for create approximate phonetics for
Tibetan text, according to Padmakara's usage.

=head1 SYNOPSIS

  use Lingua::BO::Phonetics::Padmakara;
  binmode(STDOUT, ":utf8");

  my $ph = new Lingua::BO::Phonetics::Padmakara();
  print $ph->phonetics("sems can thams cad", autosplit => 1), "\n";

=head1 DESCRIPTION

This module creates approximate phonetics for Tibetan text, according to the standards
used by Padmakara.

=cut

my %default_opts = (
  middle_nasals		=> 0,
  accent_on_e		=> 0,
  no_double_g		=> 0,
  pronounce_suffix	=> {
	b	=> 'b',
	bs	=> 'p',
	g	=> 'k',
  },

  # this together with the pronounce consonant "zl" below ensures that zla does not get devoiced to da -> ta
  pronounce_clusters	=> {
  	zl	=> 'zl',
  },

  pronounce_consonant	=> {
	kh	=> 'k',
	zl	=> 'd',			
  },
  suffix_umlaut		=> { l => undef },
  devoice_initials	=> {
  	g	=> 'k',
	d	=> 't',
	b	=> 'p',
	dr	=> 'tr',
	j	=> 'ch',
  },
  devoice_each_syllable	=> 1,
);

my %lang_opts = (
#  en => { },
#  es => {
#    pronounce_consonant	=> { 
#    	ny => "\xf1",	# use Spanish n-tilde
#	zh => 'shy',
#    },
#  },
#  fr => {
#    pronounce_umlaut	=> {
#        u  => "u",
#    },
#    pronounce_vowel	=> {
#	u  => "ou",
#    },
#  },
#  de => { },
  pt => { },
);

=head1 METHODS

=head2 CONSTRUCTOR: new (%opts)

To create a new Lingua::BO::Phonetics::Padmakara object, use this method.

Options are:

=over 4

=item - lang

What language phonetics we want.  Valid values so far are 'pt'.

=back

Any other options are passed to L<Lingua::BO::Phonetics>.

=cut

sub new {
  my ($self, %opts) = @_;
  my $default_lang = 'pt';

  my $lang = delete $opts{lang} || $default_lang;

  # return the preloaded object if we are not being passed any special options
  if (!keys(%opts) && $preloaded_object{$lang}) {
    return $preloaded_object{$lang};
  }

  # combine the global options, the language's options, and the passed options
  %opts = (%default_opts, %{ $lang_opts{$lang} }, %opts);

  # load the list of exceptions and words, if not loaded already
  load_lists() unless $lists_loaded;
  $opts{_exceptions} = $exceptions{$lang} || $exceptions{$default_lang} || {};
  $opts{_words} = \%words;

  # recycle this one from previously created Phonetics::Padmakara objects, if there were any
  $opts{_wl} = $wylie_object if $wylie_object;

  # create the object
  my $obj = $self->SUPER::new(%opts);

  # remember the language and keep the wylie object around
  $obj->{padmakara_lang} = $lang;
  $wylie_object = $obj->{_wl};

  $obj;
}

# Load the multi-lingual exceptions and the word list.  We cannot leave Lingua::BO::Phonetics
# to load the word list, because the exceptions are also words!

sub load_lists {

  # read the word list
  my $fn = Lingua::BO::Phonetics::_my_dir_file("padmakara_words.txt");
  open WO, "<:encoding(UTF-8)", $fn
    or croak "Cannot open word list $fn: $!";

  while (defined(my $l = <WO>)) {
    $l =~ s/^\s+|\s+$//g;
    $l =~ s/\x{feff}//g;
    $l =~ s/\#.*//;
    $words{$l} = 1 if $l;
  }
  close WO;

  # read the multi-lingual exceptions list
  $fn = Lingua::BO::Phonetics::_my_dir_file("padmakara_exceptions.txt");
  open EX, "<:encoding(UTF-8)", $fn
    or croak "Cannot open exceptions file $fn: $!";

  # ... read the first line with the column labels
  chomp(my $fl = <EX>);
  my ($dummy, @langs) = split /\t/, $fl;

  while (defined(my $l = <EX>)) {
    $l =~ s/^\s+|\s+$//g;
    $l =~ s/\x{feff}//g;
    $l =~ s/\#.*//;

    my ($word, @pron) = split /\t/, $l;
    next unless $word && @pron;
    $words{$word} = 1;

    for my $i (0 .. $#pron) {
      my $v = $pron[$i];
      $v =~ s/^\s+|\s+$//g;
      $exceptions{ $langs[$i] }{$word} = $v if $v;
    }
  }
  close EX;

  $lists_loaded = 1;
}


sub final_line_tweak {
  my ($self, $units) = @_;

  # "shok" at end of line should just be "sho"
  $units->[$#$units][0] =~ s/\bshok(\s*)\z/sho$1/;

  # join single syllables by twos
  my $start = 0;
  my $prev_tsekbars;
  for my $i (0 .. $#$units) {
    my $tsekbars = $units->[$i][1];
    
    # try to join a second stray syllable with the previous one
    if ($start % 2 == 1 && $tsekbars == 1 && $prev_tsekbars == 1) {
      $units->[$i - 1][0] .= $units->[$i][0];
      $units->[$i - 1][1] += $units->[$i][1];
      $units->[$i][0] = '';
      $units->[$i][1] = 0;
    }

    $start += $tsekbars;
    $prev_tsekbars = $tsekbars;
  }
}

# French needs some extra processing to get the vowels right.

sub final_tsekbar_tweak {
  my ($self, $pron) = @_;

  if ($self->{padmakara_lang} eq 'fr') {

    # u (not OU) followed by n takes an actual umlaut
    $pron =~ s/(?<!o)un/\x{fc}n/g;

    # e takes an accent, unless followed by l,m,n,k,b,p,r
    $pron =~ s/e(?![lmnkbpr])/\x{e9}/g;
  }

  $pron;
}

# Word-splitting; in addition to the word list, we also have some heuristics:
#  - single syllable + pa/ba/po/bo/mo (not "ma" as it is often a negative) makes a word
#  - "ma" + single syllable + pa/ba makes a word (ex. "ma bcos pa")
#  - single syllable + med/ldan/bral/bya/can makes a word, unless followed by pa/ba.

sub _find_word {
  my ($self, $tsek_bars, $start) = @_;

  # how many tsekbars available?
  my $max = scalar(@$tsek_bars) - $start;

  # how many tsekbars would the generic algorithm give us?
  my ($grab, $word) = $self->SUPER::_find_word($tsek_bars, $start);
  return $grab if $self->{_exceptions}{$word};

  # "ma xxx pa/ba"
  return 3
    if $max >= 3 &&
       $grab <= 3 &&
       $tsek_bars->[$start]{wylie} eq 'ma' && 
       $tsek_bars->[$start+2]{wylie} =~ /^[bp]a(?:'i|s|'am|'ang|'o|r)?$/;

  # "xxx pa/ba/po/bo/mo med/ldan/bral/bya/can" not followed by pa/ba/po/bo.
  return 3 
    if $max >= 3 &&
       $grab <= 3 &&
       $tsek_bars->[$start+1]{wylie} =~ /^(?:pa|ba|po|bo|mo)$/ &&
       $tsek_bars->[$start+2]{wylie} =~ /^(?:med|ldan|bral|bya|can)$/ &&
       ($max == 3 || $tsek_bars->[$start+3]{wylie} !~ /^[pb][ao](?:'i|s|'am|'ang|'o|r)?$/);

  # "xxx pa/ba/po/bo/mo".
  # note that we exclude "mos" from being interepreted as a weak syllable, as it is often part
  # of "mos gus" or "mos pa".
  # we also exclude "kyi bar" and "dang bar", since "bar" is here not a la-don-ified "ba".
  # same with "bar gyi", "bar chad", "bar gcod", "bar du", "bar do"
  return 2
    if $max >= 2 &&
       $grab <= 2 &&
       $tsek_bars->[$start+1]{wylie} =~ /^(?:pa|ba|po|bo|mo(?!s))(?:'i|s|'am|'ang|'o|r)?$/ &&
       ($tsek_bars->[$start+1]{wylie} ne 'bar' || 
         ($tsek_bars->[$start]{wylie} !~ /^(?:dang|kyi|gyi|yi|gi)$/ &&
	  ($max == 2 || $tsek_bars->[$start+2]{wylie} !~ /^(?:gyi|chad|gcod|du|do)$/)));

  # "xxx med/ldan/bral/bya/can" not followed by pa/ba/po/bo.  (xxx cannot be "dang")
  return 2
    if $max >= 2 &&
       $grab <= 2 &&
       $tsek_bars->[$start+1]{wylie} =~ /^(?:med|ldan|bral|bya|can)$/ &&
       $tsek_bars->[$start]{wylie} !~ /^(?:dang|su)$/ &&
       ($max == 2 || $tsek_bars->[$start+2]{wylie} !~ /^[pb][ao](?:'i|s|'am|'ang|'o|r)?$/);

  return $grab;
}

=head1 MODULE INITIALIZATION

Creates a Lingua::BO::Phonetics::Padmakara object at startup, to preload the word lists and
initialize the Wylie converter once and for all.

=cut

{
  $preloaded_object{pt} = Lingua::BO::Phonetics::Padmakara->new(lang => 'pt') || 
    croak "Cannot create initial Lingua::BO::Phonetics::Padmakara object";
}

1;


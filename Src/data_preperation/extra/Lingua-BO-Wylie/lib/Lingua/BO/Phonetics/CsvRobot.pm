package Lingua::BO::Phonetics::CsvRobot;

use strict;
use utf8;

use Lingua::BO::Phonetics::Rigpa;

my %languages = (
  en => "english",
  es => "spanish",
  fr => "french",
  de => "german",
);

sub try_do_csv {
  my ($txt, $lang, $spanish_checking) = @_;
  $spanish_checking = ($lang eq 'es');

  die "Cannot do this language($lang)" unless $languages{$lang};

  my $pho = new Lingua::BO::Phonetics::Rigpa(lang => $lang);

  my @lines = split /\r?\n|\r/, $txt;

  my $fl = get_line(\@lines);
  my @fl = split_line($fl);

  # figure out where new columns go
  my %lang_cols;
  my %lang_old;

  die "First column should start with 'type'.\n" unless $fl[0] =~ /^type/i;
  my ($tibcol) = grep { $fl[$_] =~ /^tibetan$/i } 1 .. $#fl;
  die "Tibetan column not found.\n" unless defined $tibcol;

  {
    my $lang_name = $languages{$lang};

    # do we already have a column with that language phonetics?
    my ($col) = grep { $fl[$_] =~ /^$lang_name\s+phonetics$/i } 1 .. $#fl;

    # ... else stick it at the end of the "phonetics" bits for spanish, at the beginning for english
    my $col2;
    if (!$col) {
      my @phocols = grep { $fl[$_] =~ /\s+phonetics$/i } 1 .. $#fl;
      if (@phocols) {
	if ($lang eq 'en') {
	  $col2 = $phocols[0];
	} elsif ($lang eq 'es') {
	  $col2 = $phocols[$#phocols] + 1;
	}
      }
    }

    if ($col) {
      $lang_cols{$lang} = $col;
      $lang_old{$lang} = $col;
      $fl[$col] = "$lang_name phonetics (not checked)";
      splice @fl, $col+1, 0, "$lang_name phonetics (old)";

    } elsif ($col2) {
      $lang_cols{$lang} = $col2;
      splice @fl, $col2, 0, "$lang_name phonetics (not checked)";
    
    } else {
      $lang_cols{$lang} = $#fl + 1;
      push @fl, "$lang_name phonetics (not checked)";
    }
  }

  # spanish checking - locate the English phonetics
  my ($eng_col) = grep { $fl[$_] =~ /^english\s+phonetics$/i } 1 .. $#fl;

  my @out;

  # spit out the new header line (no BOM)
  push @out, join("\t", map { qq/"$_"/ } @fl) . "\r\n";

  # do the lines
  while (my $l = get_line(\@lines)) {

    my @l = split_line($l);

    # put the languages in
    my $tib = $l[$tibcol];
    my $type = $l[0];
    my $phon;
    if ($type =~ /verse|mantra/i) {
      $phon = $pho->phonetics($tib, autosplit => 1);

      # automatic consistency check of spanish based on english
      if ($lang eq 'es' && $spanish_checking && $eng_col) {
	my $eng = $l[$eng_col];
	my $spa = $phon;

	$eng =~ s/\s+/ /g;
	$eng =~ s/^ +| +$//g;

	$eng =~ s/(^|\s)j/$1ch/g;	# j into ch at the beginning of words
	$spa =~ s/(^|\s)j/$1ch/g;

	$eng =~ s/é/e/g;		# é into e

	$eng =~ s/ny/ñ/g;		# ny into ñ
	$spa =~ s/ny/ñ/g;

        $eng =~ s/ngg/ng/g;		# wanggyur -> wangyur
        $spa =~ s/ngg/ng/g;

	$eng =~ s/ig($|\s)/ik$1/g;	# rangrig -> rangrik
	$spa =~ s/ig($|\s)/ik$1/g;
	$eng =~ s/chenrezi(?:g|k)/chenresik/;	# chenresik

	$eng =~ s/\s+/ /g;		# collapse multiple spaces in the english
	$eng =~ s/^\s+|\s+$//gs;	# trim spaces at the beginning or end

	$eng =~ s/ṃ/m/g;		# skt diacritics begone
	$eng =~ s/ā/a/g;
	$eng =~ s/ī/i/g;
	$eng =~ s/ū/u/g;
	$eng =~ s/ḥ/h/g;
	$eng =~ s/ś/sh/g;
	$eng =~ s/ṣ/sh/g;

	$spa =~ s/\(\?\)//g;		# remove "questionable" marks (we already add our own >>)

	unless ($eng eq $spa) {
	  $phon = ">>$phon";
	  print STDERR "ENG [[$eng]] SPA [[$spa]]\n" if -t STDERR;
	}
      }

    } else {
      # $phon = $l[ $lang_old{$lang} ];
      $phon = '';
    }

    splice @l, $lang_cols{$lang}, 0, $phon;

    # spit out the line
    push @out, join("\t", map { qq/"$_"/ } @l) . "\r\n";
  }

  return join '', @out;
}

sub get_line {
  my $ls = shift;
  my $l = shift @$ls;

  # as long as we have an odd number of quotes, append the next line
  while ($l =~ /^[^"]*(?:"[^"]*"[^"]*)*"[^"]*$/) {
    my $nx = shift @$ls;
    last unless defined $nx;
    $l .= $nx;
  }

  $l;
}

sub split_line {
  my $l = shift;

  $l =~ s/[\r\n]+$//;
  $l =~ s/\x{feff}//g;
  my @l = split /\t/, $l, -1;
  map { s/^"+|"+$//g; } @l;

  return @l;
}

1;


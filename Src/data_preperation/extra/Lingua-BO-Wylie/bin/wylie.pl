#!/usr/bin/perl

use strict;
BEGIN { 
  push @INC, '../lib' if -f '../lib/Lingua/BO/Wylie.pm';
  push @INC, './lib' if -f './lib/Lingua/BO/Wylie.pm';
  push @INC, '/srv/www/perl-lib' if -f '/srv/www/perl-lib/Lingua/BO/Wylie.pm';
}

use utf8;
use Lingua::BO::Wylie;
use Getopt::Long;

my ($repeat, $help, $unicode, $skt);
GetOptions(
  "unicode"	=> \$unicode,
  "repeat=s"	=> \$repeat,
  "help"	=> \$help,
  "skt"		=> \$skt,
);

sub help {
  print "This is the command-line interface to the Wylie conversion module.

Use: $0 [options] inputfile outputfile

Converts between Wylie and Unicode, and vice-versa.  All Unicode uses the
UTF-8 encoding.

Options are:
  -u             - convert from Unicode to Wylie.  Otherwise, does the 
                   opposite conversion.
  -r '//'        - also reprint the original before the converted version, 
                   separated by '//' (or whatever)
  --skt          - use Sanskrit diacritics instead of standard Wylie (not 100% perfect)
  -h             - get this help

";
  exit 0;
}

help() if $help;

my $wl = new Lingua::BO::Wylie(
  check_strict => 1,
  print_warnings => 0
);

my ($infile, $outfile) = @ARGV[0, 1];

help() if !defined($infile) || !defined($outfile);

if ($infile eq '-') {
  *IN = *STDIN;
} else {
  open IN, "<", $infile or die "Cannot read $infile.";
}
if ($outfile eq '-') {
  *OUT = *STDOUT;
} else {
  open OUT, ">", $outfile or die "Cannot write to $outfile.";
}
binmode(IN, ":utf8");
binmode(OUT, ":utf8");

while(defined(my $in = <IN>)) {
  my $out = $unicode ? $wl->to_wylie($in) : $wl->from_wylie($in);
  $out = sanskritize($out) if $skt;
  if ($repeat) {
    chomp($in);
    print OUT "$in\t$out";
  } else {
    print OUT "$out";
  }
}

sub sanskritize {
  my $in = shift;
  $in =~ s/ny/ñ/g;
  $in =~ s/A/ā/g;
  $in =~ s/I/ī/g;
  $in =~ s/U/ū/g;
  $in =~ s/N/ṇ/g;
  $in =~ s/T/ṭ/g;
  $in =~ s/D/ḍ/g;
  $in =~ s/\~M`/ṃ/g;
  $in =~ s/\~M/ṃ/g;
  $in =~ s/M/ṃ/g;
  $in =~ s/H/ḥ /g;
  $in =~ s/dz/j/g;
  $in =~ s/tsh/ch/g;
  $in =~ s/ts/c/g;
  #$in =~ s/b/v/g;
  $in =~ s/w/v/g;
  $in =~ s/Sh/ṣ/g;
  $in =~ s/sh/ś/g;
  $in =~ s/r-i/ṛ/g;
  $in =~ s/r-I/ṝ/g;
  $in =~ s/\+//g;
  $in =~ s/\bba ?jra/vajra/g;
  $in =~ s/\bsarba/sarva/g;
  $in =~ s/_/ /g;
  $in =~ s/ +/ /g;
  $in =~ s/: *$//g;

  $in =~ s/\/ *$//g;
  $in =~ s/ *$//;
  $in =~ s/^ *//;
  $in =~ s/\/ / \| /g;

  $in;
}


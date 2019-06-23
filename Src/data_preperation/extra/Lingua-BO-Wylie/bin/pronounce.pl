#!/usr/bin/env perl

use strict;
BEGIN { 
  push @INC, '../lib' if -f '../lib/Lingua/BO/Phonetics.pm';
  push @INC, './lib' if -f './lib/Lingua/BO/Phonetics.pm';
  push @INC, '/srv/www/perl-lib' if -f '/srv/www/perl-lib/Lingua/BO/Wylie.pm';
}

use Lingua::BO::Phonetics;
use Getopt::Long;

my ($style, $repeat, $joiner, $separator, $autosplit, $caps, $help);
GetOptions(
  "style=s"	=> \$style,
  "repeat=s"	=> \$repeat,
  "joiner=s"	=> \$joiner,
  "separator=s"	=> \$separator,
  "autosplit"	=> \$autosplit,
  "caps=s"	=> \$caps,
  "help"	=> \$help,
);

sub help {
  print "This is the command-line interface to the Tibetan Phonemic converter.

Use: $0 [options] inputfile outputfile

The input file can be in Wylie or Tibetan Unicode.  If it is Unicode, it should
be encoded in UTF-8.

Options are:
  -sty style-name  - which phonemic system to use; the default is rigpa-en.
                   Other styles are 'thl', 'rigpa-es', 'rigpa-fr', 'rigpa-de'.
  -r '//'        - also reprint the tibetan before the pronounciation, 
                   separated by '//' (or whatever)
  -j '-'         - use '-' (or whatever character) to mark syllables belonging 
                   to a single word
  -sep ' '       - use ' ' (or whatever) to separate syllables belonging to 
                   different words
  -a             - auto-split words based on a small built-in dictionary
  -c '*'         - use '*' (or whatever) to mark words to be capitalized
  -h             - get this help

Only one of -j, -sep or -a can be given.  Default is auto-split.
";
  exit 0;
}

help() if $help;

if (defined($joiner) + defined($separator) + defined($autosplit) > 1) {
  die "At most one of -j, -sep or -a can be given.\n";
}
my %pho_args;
if (defined($joiner)) {
  $pho_args{joiner} = $joiner;
} elsif (defined($separator)) {
  $pho_args{separator} = $separator;
} else {
  $pho_args{autosplit} = 1;
}

my %new_args = (print_warnings => 0);
my $class = 'Lingua::BO::Phonetics';

$style ||= 'rigpa-en';

if ($style =~ /^rigpa-(\w\w)$/) {
  my $lang = $1;
  $class = 'Lingua::BO::Phonetics::Rigpa';
  eval "require $class"; die $@ if $@;
  $new_args{lang} = $lang;

} elsif ($style =~ /^padmakara-(\w\w)$/) {
  my $lang = $1;
  $class = 'Lingua::BO::Phonetics::Padmakara';
  eval "require $class"; die $@ if $@;
  $new_args{lang} = $lang;
  die "Padmakara is only supported in Portuguese for now.\n" unless $lang eq 'pt';
}

my $pho = $class->new(%new_args);
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
  my $out = $pho->phonetics($in, %pho_args);
  if ($repeat) {
    chomp($in);
    print OUT "$in\t$out";
  } else {
    print OUT "$out";
  }
}


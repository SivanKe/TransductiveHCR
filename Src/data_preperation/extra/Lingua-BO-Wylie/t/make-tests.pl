#!/usr/bin/perl

use strict;
use lib '../lib';
use Lingua::BO::Wylie;

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

my $w = new Lingua::BO::Wylie(check_strict => 0, print_warnings => 1);
my $w2 = new Lingua::BO::Wylie(print_warnings => 1);

print "# wylie\tunicode\twarnings\twylie back\twarnings\tdifferent unicode?\n";

while (<>) {
  chomp;
  my $str = $_;
  my $s = $w->from_wylie($str);
  my $e = $w->get_warnings();

  my $s2 = $w2->from_wylie($str);
  my $e2 = $w2->get_warnings();

  my $rewylie = $w->to_wylie($s);
  my $e3 = $w->get_warnings();
  my $reuni = $w->from_wylie($rewylie);

  die "$str: different outputs! ($s) ($s2)" unless $s eq $s2;
  die "$str: warnings on normal but not on strict!" if @$e && !@$e2;

  my @out = ($str);
  push @out, $s;
  if (@$e2 && !@$e) {
    push @out, "1";
  } elsif (@$e) {
    push @out, "2";
  } else {
    push @out, 0;
  }
  push @out, "$rewylie";
  push @out, scalar(@$e3);
  push @out, $reuni eq $s ? "0" : "1";
  print join("\t", @out), "\n";
}


#!/usr/bin/perl

use strict;
use lib '/srv/www/perl-lib';
use Encode ();
use Lingua::BO::Wylie;

use CGI ();
$CGI::POST_MAX = 1024 * 1024;	# max 1MB posts
binmode(STDOUT, ":utf8");

my %lists = (
  rigpa_words		=> 'rigpa_words.txt',
  rigpa_exceptions	=> 'rigpa_exceptions.txt',
  thl_words		=> 'word_list.txt',
  thl_exceptions	=> 'exceptions.txt',
);

my $list = CGI::param('list');
$list = 'rigpa_exceptions' unless $lists{$list};
my $fn  = $lists{$list};


my $pho;
if ($list =~ /except/) {
  # nothing
} elsif ($list =~ /rigpa/) {
  require Lingua::BO::Phonetics::Rigpa;
  $pho = new Lingua::BO::Phonetics::Rigpa(print_warnigns => 0);
} else {
  require Lingua::BO::Phonetics;
  $pho = new Lingua::BO::Phonetics(print_warnigns => 0);
}

my $path = $INC{"Lingua/BO/Wylie.pm"} || die "Cannot find load path for Lingua::BO::Wylie.";  
$path =~ s/\/Wylie.pm$//;
my $fp = $path . "/" . $fn;
die "Cannot find file '$fp'." unless -f $fp;

my @words;
READ: {
  open FH, "<:encoding(UTF-8)", $fp or last READ;
  while (defined($_ = <FH>)) {
    s/^\s+|\s+$//gs;
    s/\#.*//;
    next unless /\w/;
    my ($word, $pron) = split /\t/, $_, 2;
    if ($pho) {
      $pron = $pho->phonetics($word, separator => '!@!@!');
    } else {
      $pron =~ s/\t/ \/ /g;
    }
    push @words, [ $word, $pron ];
  }
  close FH;
}

my $automatic = '';

if ($pho) {
  $automatic = '(automatic)';
} elsif ($list eq 'rigpa_exceptions') {
  $automatic = '(english / french / spanish / german)';
}

# HTML output
print CGI::header(-charset => "utf-8");

print <<_HTML_;
<html>
<head>
<meta name="robots" content="noindex, nofollow">
<style>
  body, td { background: #fff; font-family: tahoma; font-size: 12px; }
</style>
<title>Word lists for Robot Tibetan Phonetics</title>
</head>

<body><center>
<big><b>Word list for Robot Tibetan Phonetics: $list</b></big>
<br><br>
<table cellspacing="2" cellpadding="2" border="0">
<tr>
<th align="left">Word</th>
<th align="left">Pronounciation $automatic</th>
</tr>
_HTML_

foreach my $w (sort { $a->[0] cmp $b->[0] } @words) {
  print "<tr><td>" . CGI::escapeHTML($w->[0]) . "</td><td>" . CGI::escapeHTML($w->[1]) . "</td></tr>\n";
}

print <<_HTML_;
</table></center>
</body></html>
_HTML_


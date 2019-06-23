#!/usr/bin/perl

use strict;
use lib '/srv/www/perl-lib';
use Lingua::BO::Phonetics::CsvRobot;
use Encode ();

use CGI ();
$CGI::POST_MAX = 1024 * 1024;	# max 1MB posts

binmode(STDOUT, ":utf8");

my $lang = CGI::param("lang") || "en";

# perl-generate the selects so we can keep the values on page reload
my %selects = (
  lang	=> [
  	[ "en",	"Rigpa Phonetics - ENGLISH" ],
  	[ "fr",	"Rigpa Phonetics - FRENCH" ],
  	[ "es",	"Rigpa Phonetics - SPANISH" ],
  	[ "de",	"Rigpa Phonetics - GERMAN" ],
  ],
);

sub make_options {
  my $name = shift;
  my @out;
  my $opts = $selects{$name};
  foreach my $opt (@$opts) {
    my ($value, $label);
    if (ref($opt)) {
      ($value, $label) = @$opt;
    } else {
      $value = $label = $opt;
    }
    my $sel = ((CGI::param($name) || '') eq $value) ? 'selected' : '';
    push @out, qq{<option $sel value="$value">$label</option>\n};
  }
  return join '', @out;
}


my %html_options = map { $_ => make_options($_) } keys %selects;
my $error;

my $upl = CGI::upload('csv_file');
if ($upl) {

  eval {
    binmode($upl, ":encoding(UTF-8)");
    local $/ = undef;
    my $txt = <$upl>;

    my $fn = CGI::param('csv_file');
    $fn =~ s/.*[\/\\]//g;
    $fn ||= 'file.csv';
    $fn .= ".csv" unless $fn =~ /\.csv$/i;
    $fn =~ s/"//g;

    $txt = Lingua::BO::Phonetics::CsvRobot::try_do_csv($txt, $lang);
    print CGI::header(
      -type       => "text/csv",
      -charset    => "utf-8",
      -attachment => $fn,
    );
    print $txt;
    exit 0;
  };

  $error = $@;
}

# HTML output
print CGI::header(-charset => "utf-8");

if ($error) {
  $error = qq{<br><br><font color="#ff0000">Error: } . CGI::escapeHTML($error) . "</font><br>";
}

print <<_HTML_;
<html>
<head>
<style>
  body { background: #fff; margin-left: 15px; }
  body, td, input, select, textarea { font-family:verdana, tahoma, helvetica; font-size: 12px; }
  .warn { font-family: tahoma; font-size: 12px; }
  .eng { font-family: tahoma; font-size: 14px; }
  .after { font-size: 10px; }
  .title { font-size: 16px; font-weight: bold; }
</style>
<title>Robot Tibetan Phonetics for CSV files</title>
</head>
<body>
<form id="id__form" method="POST" action="/cgi-bin/csv.pl/prayer.csv?lang=$lang" enctype="multipart/form-data">
<span class="title">Robot Tibetan Phonetics for CSV files</span><br><br>

<div style="width: 80%">
<select id="id__type" name="lang">
$html_options{lang}
</select>

<br>
Please upload a CSV file: 
<input type="file" name="csv_file" size="25"> <input type="submit" name="send" value="Go">

<br><br>
<span style="color: #c00;">Please note that automatically generated phonetics cannot possibly be 100% accurate, because the way
words are separated in Tibetan is inherently ambiguous. It is important that someone who understands
the Tibetan text checks the phonetics.</span>
<br>
$error
</div>

<div class="after">
<br><br>
&bull; This conversion code is Free Software; you can <a href="/tibetan/Lingua-BO-Wylie-dev.zip">download the Perl module here</a>.

</div>

<br><br>

</body>
</html>
_HTML_


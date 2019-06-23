#!/usr/bin/perl

use strict;
use lib '/srv/www/perl-lib';
use Encode ();

use CGI ();
$CGI::POST_MAX = 1024 * 1024;	# max 1MB posts
$CGI::DISABLE_UPLOADS = 1;  	# no uploads

binmode(STDOUT, ":utf8");

my $input = CGI::param("input");
$input = '' unless defined $input;
$input = Encode::decode_utf8($input);

my $sep_join = CGI::param("sep_join");
$sep_join = '' unless defined $sep_join;
$sep_join = Encode::decode_utf8($sep_join);
my $sep_join_enc = CGI::escapeHTML($sep_join);

my $words = CGI::param("words") || "none";
my $type = CGI::param("type") || "thl";

my $show = $input;

# perl-generate the selects so we can keep the values on page reload
my %selects = (
  words	=> [
	[ "auto", 	"Use built-in word list" ],
  	[ "none",	"Handle each syllable separately" ],
	[ "join", 	"Syllables in a word joined by" ],
	[ "sep", 	"Words separated by" ],
  ],
  type	=> [
  	[ "thl",	"THL Simplified Phonemic Transcription" ],
  	[ "rigpa-en",	"Rigpa Phonetics - ENGLISH" ],
  	[ "rigpa-fr",	"Rigpa Phonetics - FRENCH" ],
  	[ "rigpa-es",	"Rigpa Phonetics - SPANISH" ],
  	[ "rigpa-de",	"Rigpa Phonetics - GERMAN" ],
  	[ "padmakara-pt",	"Padmakara PT - Int. Simplified Phonetics" ],
  ],
);

sub trim {
  my $string = shift;
  $string =~ s/^\s+|\s+$//gs;
  $string;
}

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

my %pho_args = ();
if ($words eq 'auto') {
  $pho_args{autosplit} = 1;
}  elsif ($words eq 'sep' && $sep_join ne '') {
  $pho_args{separator} = $sep_join;
}  elsif ($words eq 'join' && $sep_join ne '') {
  $pho_args{joiner} = $sep_join;
}

my $class = 'Lingua::BO::Phonetics';
my %class_args;

if ($type =~ /^rigpa-(\w\w)$/) {
  $class_args{lang} = $1;
  $class = "Lingua::BO::Phonetics::Rigpa";

} elsif ($type =~ /^padmakara-(\w\w)$/) {
  die "Padmakara is only supported in Portuguese for now.\n" unless $1 eq 'pt';
  $class_args{lang} = $1;
  $class = 'Lingua::BO::Phonetics::Padmakara';
}

eval "require $class"; die $@ if $@;

my %html_options = map { $_ => make_options($_) } keys %selects;

$input =~ s/(?:\r|\n)*$//;
$input =~ s/\r\n|\r|\n/\n/g;

my $out = '';
my $warns;
my $checked = '';

if ($input ne '') {
  my $pho = $class->new(print_warnings => 0, %class_args);
  $out = $pho->phonetics($input, %pho_args);
  $warns = $pho->get_warnings();

  # "add input to output" code by Martin Dudek <goosebumps4all@gmail.com>
  if (CGI::param('addInput')) {
    $checked = "checked='checked'";
    my @valuesinput = split "\n", $input;
    my @valuesoutput = split "\n", $out;
    $out = '';

    for my $j (0 .. $#valuesinput) {
      my $in = trim($valuesinput[$j]);
      $out .= $in . "\n";
      $out .= trim($valuesoutput[$j]) . "\n" if $in ne "";
    }
  }
}

# plain text output?
if (CGI::param('plain')) {
  print CGI::header(-charset => "utf-8", -type => "text/plain");
  print $out;
  if ($warns && @$warns) {
    print "\n\n---\n";
    foreach my $w (@$warns) {
      print "$w\n";
    }
  }
  exit 0;
}

# HTML output
print CGI::header(-charset => "utf-8");

print <<_HTML_;
<html>
<head>
<style>
  body { background: #fff; margin-left: 15px; }
  body, td, input, select, textarea { font-family:verdana, tahoma, helvetica; font-size: 12px; }
  .tib { font-family: "Tibetan Machine Uni"; font-size: 28px; }
  .warn { font-family: tahoma; font-size: 12px; }
  .eng { font-family: tahoma; font-size: 14px; }
  .after { font-size: 10px; }
  .title { font-size: 16px; font-weight: bold; }
  textarea#id__txt { margin-top: 5px; margin-bottom: 5px; padding: 5px; border: 1px solid blue; height: 120px; width: 85%; font-family: fixed; font-size: 14px; }
</style>
<title>Tibetan Phonetics</title>
</head>
<body>
<form id="id__form" method="POST">
<span class="title">Tibetan Phonetics</span><br><br>

<div style="width: 80%">
<table width="100%" cellspacing="0" cellpadding="0"><tr>
<td width="60%" align="left" style="white-space: nowrap;">
Paste your Tibetan text below in Unicode or Wylie, and click "Make Phonetics" or press Ctrl+Enter.
</td>
</tr></table></div>

<textarea id="id__txt" style="font-size: 14px;" onkeydown="return textarea_keydown(event);" name="input">$show</textarea><br>
<div style="width: 80%">
<table width="100%" cellspacing="0" cellpadding="0" border="0">
<tr>
<td width="20%" align="right">
Word&nbsp;splitting:&nbsp;
</td>
<td width="40%" align="left" style="white-space: nowrap;">
<select id="id__words" name="words" onchange="javascript:change_words();">
$html_options{words}
</select>
<input style="display: none;" type="text" size="2" maxlength="3" id="id__sep_join" name="sep_join" value="$sep_join_enc">
</td>
<td width="40%" align="center">
<input type="submit" name="send" value="Make Phonetics" onclick="javascript:return check_form();">
</td>
</tr>
<tr>
<td align="right">
Phonemic&nbsp;system:&nbsp;
</td>
<td align="left">
<select id="id__type" name="type">
$html_options{type}
</select>
</td>
</tr>
<tr>
<td align="right"><input type="checkbox" name="plain" value="y">&nbsp;</td>
<td align="left">Plain text output</td>
</tr>
<tr>
<td align="right"><input type="checkbox" name="addInput" value="y" $checked >&nbsp;</td>
<td align="left">Add input to output</td>
</tr>
</table>
<br>
<span style="color: #c00;">Please note that automatically generated phonetics cannot possibly be 100% accurate, because the way
words are separated in Tibetan is inherently ambiguous. It is important that someone who understands
the Tibetan text checks the phonetics.</span>
<br>
</div>
</form>
<br>

_HTML_

if ($out ne '') {
  $out = CGI::escapeHTML($out);

  print <<_HTML_;
Tibetan Phonetics:<br>
<textarea style="border: 1px solid #888; background: #eef; padding: 8px; width: 85%; height: 250px;" class="eng">$out</textarea><br>
_HTML_
}

if ($warns && @$warns) {
  $warns = join "<br>\n", map { CGI::escapeHTML($_) } @$warns;
  print <<_HTML_;
<br>
Warnings:
<div style="border: 1px solid #888; background: #ff8; padding: 8px; width: 85%" class="warn">
$warns
</div>
_HTML_
}

finish();

sub finish {
  print <<_HTML_;
<div class="after">
<br><br>
&bull; This conversion code is Free Software; you can <a href="/tibetan/Lingua-BO-Wylie-dev.zip">download the Perl module here</a>.

<br>
&bull; See the definition of <a target="_blank" href="http://www.thlib.org/reference/transliteration/#essay=/thl/phonetics/">THL 
Simplified Tibetan Phonemic Transcription</a>

<br>
&bull; View the <a href="/cgi-bin/view-list.pl?list=rigpa_words" rel="nofollow">word list</a> and the <a href="/cgi-bin/view-list.pl?list=rigpa_exceptions" rel="nofollow">exceptions list</a> for Rigpa English Phonetics.
</div>

<br><br>
<script language="javascript">
function textarea_keydown(e) {
  // submit on control-Enter
  if (e && e.keyCode == 13 && e.ctrlKey) {
    document.getElementById('id__form').submit();
    return false;
  }
  return true;
}

function change_words() {
  var e = document.getElementById('id__words');
  var wds = e.options[e.selectedIndex].value;
  if (wds == 'sep' || wds == 'join') {
    document.getElementById('id__sep_join').style.display="inline";
  } else {
    document.getElementById('id__sep_join').style.display="none";
  }
}

function check_form() {
  var e = document.getElementById('id__words');
  var wds = e.options[e.selectedIndex].value;
  var sj = document.getElementById('id__sep_join').value;
  if (sj == '') {
    if (wds == 'sep') {
      alert("Please specify the character or string used to separate words.");
      var sj = document.getElementById('id__sep_join').select();
      return false;
    } else if (wds == 'join') {
      alert("Please specify the character or string used to join syllables in the same word.");
      var sj = document.getElementById('id__sep_join').select();
      return false;
    }
  }
  return true;
}

document.getElementById('id__txt').select();
document.getElementById('id__txt').focus();
change_words();

</script>

</body>
</html>
_HTML_
}


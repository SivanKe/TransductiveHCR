# Before `make install' is performed this script should be runnable with
# `make test'. After `make install' it should work as `perl Lingua-BO-Wylie.t'

#
# TO RUN THE TESTS, from the main dir:
# perl -Ilib t/Lingua-BO-Wylie.t

#########################

# can't be bothered to count the tests!
use Test::More qw(no_plan);
BEGIN { use_ok('Lingua::BO::Wylie') };

#########################

# Insert your test code below, the Test::More module is use()ed here so read
# its man page ( perldoc Test::More ) for help writing this test script.

#########################

# Create a couple objects, one with more warnings than the other
my $w = new Lingua::BO::Wylie(check_strict => 0, print_warnings => 0);
my $w2 = new Lingua::BO::Wylie(print_warnings => 0);

ok($w, 'Create Lingua::BO::Wylie object');
ok($w2, 'Create 2nd Lingua::BO::Wylie object');

# read the test lines from a file...
open IN, "t/test.txt" or die "Open: test.txt: $!";
binmode(IN, ":utf8");

while (my $line = <IN>) {
  chomp $line;
  next if $line =~ /^\#/;
  my ($wylie, $uni, $warns, $wylie2, $wylie_warns, $uni_diff) = split /\t/, $line;

  # convert with the two encoders (w/ more & less warnings)
  my $s = $w->from_wylie($wylie);
  my $e = $w->get_warnings();

  my $s2 = $w2->from_wylie($wylie);
  my $e2 = $w2->get_warnings();

  my $d1 = $w->_dump($s);
  my $d2 = $w->_dump($uni);

  my $pw = $wylie;
  $pw =~ s/[\x{100}-\x{ffff}]/{unichar}/g;

  # re-encode into wylie
  my $rewylie = $w->to_wylie($s);
  my $e3 = $w->get_warnings();

  # and again back into unicode
  my $reuni = $w->from_wylie($rewylie);

  # the two first unicodes must be same (only diff. is generating more or less warnings)
  ok($s eq $s2, "[$pw]: same encodings w/ and w/o strict checking.");

  # expected unicode?
  ok($s eq $uni, "[$pw]: correct unicode [$d1] [$d2].");

  # expected warnings?
  ok(!@$e || @$e2, "[$pw]: strict warnings");
  
  if ($warns == 0) {
    ok(!@$e && !@$e2, "[$pw]: no warnings.");
  } elsif ($warns == 1) {
    ok(!@$e, "[$pw]: no non-strict warnings.");
    ok(@$e2, "[$pw]: have strict warnings.");
  } elsif ($warns == 2) {
    ok(@$e, "[$pw]: have warnings.");
  }

  # expected re-encoded wylie?
  ok($wylie2 eq $rewylie, "[$pw]: to_wylie expected [$wylie2], got [$rewylie].");

  # expected warnings in re-encoding to wylie?
  if ($wylie_warns == 0) {
    ok(!@$e3, "[$pw]: no to_wylie warnings.");
  } else {
    ok(@$e3, "[$pw]: to_wylie warnings.");
  }

  # expected re-encoded unicode (unless it's supposed to be different)
  ok($reuni eq $s, "[$pw]: round-trip to unicode.") unless $uni_diff;
  ok($reuni ne $s, "[$pw]: should not round-trip to unicode.") if $uni_diff;
}

1;

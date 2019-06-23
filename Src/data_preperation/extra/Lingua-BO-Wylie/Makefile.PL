use 5.008008;
use ExtUtils::MakeMaker;
# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.
WriteMakefile(
    NAME              => 'Lingua::BO::Wylie',
    VERSION_FROM      => 'lib/Lingua/BO/Wylie.pm', # finds $VERSION
    EXE_FILES	      => [ 'bin/wylie.pl', 'bin/pronounce.pl' ],
    PREREQ_PM         => { 'File::Spec' => 1, 'Carp' => 1 }, # e.g., Module::Name => 1.1
    ($] >= 5.005 ?     ## Add these new keywords supported since 5.005
      (ABSTRACT_FROM  => 'lib/Lingua/BO/Wylie.pm', # retrieve abstract from module
       AUTHOR         => 'Roger Espel Llima <rogerespel@yahoo.com>') : ()),
);

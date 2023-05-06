{ stdenv, cmake, some-lib }:

stdenv.mkDerivation {
  pname = "some-app";
  version = "0.0.1";
  src = ./.;

  nativeBuildInputs = [ cmake ];
  buildInputs = [ some-lib ];
}


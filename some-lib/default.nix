{ stdenv, cmake }:

stdenv.mkDerivation {
  pname = "some-lib";
  version = "0.0.1";
  src = ./.;

  nativeBuildInputs = [ cmake ];

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
  ];
}

{ stdenv
, cmake
, some-lib
, lddHook
}:

stdenv.mkDerivation {
  pname = "some-app";
  version = "0.0.1";
  src = ./.;

  nativeBuildInputs = [
    cmake
    lddHook
  ];
  buildInputs = [ some-lib ];

  lddForcePrint = true;
}


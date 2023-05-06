{ stdenv
, cmake
, lddHook
}:

stdenv.mkDerivation {
  pname = "some-lib";
  version = "0.0.1";
  src = ./.;

  nativeBuildInputs = [
    cmake
    lddHook
  ];

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
  ];
}

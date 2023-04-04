with import <nixpkgs>
{
  config.allowUnfree = true;
  config.cudaSupport = true;
  config.cudaCapabilities = [ "8.6" ];
};

rec {
  use-cudatoolkit-root = callPackage ./use-cudatoolkit-root.nix { };
  use-legacy = callPackage ./use-legacy.nix { };

  shell =
    mkShell rec {
      inputsFrom = [ use-cudatoolkit-root ];

      cudaFlags = [
        "-DCUDAToolkit_ROOT=${use-cudatoolkit-root.CUDAToolkit_ROOT}"
      ];
    };
}


{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/master";
  description = "Make consuming FindCUDAToolkit.cmake easier with Nix";

  outputs = { self, nixpkgs }: {

    packages.x86_64-linux =
      let
        system = "x86_64-linux";
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
          config.cudaCapabilities = [ "8.6" ];
        };
        inherit (pkgs) callPackage;
        selfPackages = self.packages.${system};
      in
      {
        use-cudatoolkit-root = callPackage ./use-cudatoolkit-root.nix {
          stdenv = pkgs.cudaPackages.backendStdenv;
          inherit (selfPackages) lddHook;
        };
        use-cudatoolkit-root-wrong-stdenv = callPackage ./use-cudatoolkit-root.nix {
          inherit (selfPackages) lddHook;
        };
        use-legacy = callPackage ./use-legacy.nix {
          inherit (selfPackages) lddHook;
        };
        some-lib = callPackage ./some-lib {
          inherit (selfPackages) lddHook;
        };
        some-app = callPackage ./some-app {
          inherit (selfPackages) some-lib lddHook;
        };
        lddHook = pkgs.makeSetupHook
          {
            name = "ldd-hook";
            substitutions = {
              ldd = "${pkgs.lib.getBin pkgs.pkgsBuildHost.glibc}/bin/ldd";
              patchelf = "${pkgs.pkgsBuildHost.patchelf}/bin/patchelf";
            };
          } ./ldd-hook.sh;

        default = selfPackages.use-cudatoolkit-root;
      };
  };
}

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
      in
      {
        use-cudatoolkit-root = callPackage ./use-cudatoolkit-root.nix { };
        use-legacy = callPackage ./use-legacy.nix { };
      };
  };
}

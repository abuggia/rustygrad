{
  description = "Starcast development support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      overlays = [ (import rust-overlay) ];
      system = "aarch64-darwin";
      pkgs = import nixpkgs {
        inherit overlays system;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          rust-bin.beta.latest.default
          just
          cargo-watch
          rust-analyzer
        ];
      };
    };
}

{
  description = "mbf-singlecell-plotter- application using uv2nix";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      my_fixes = final: prev: {
        dppd-plotnine = prev.dppd-plotnine.overrideAttrs (old: {
          nativeBuildInputs = old.nativeBuildInputs ++ (final.resolveBuildSystem { setuptools = [ ]; });
        });
        dppd = prev.dppd.overrideAttrs (old: {
          nativeBuildInputs = old.nativeBuildInputs ++ (final.resolveBuildSystem { setuptools = [ ]; });
        });
      };

      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python314;
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
              my_fixes
            ]
          )
      );

    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system}.overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "mbf-singlecell-plotter-ev-env" workspace.deps.all;
        in
        {
          default = pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
            '';
          };
        }
      );

      packages = forAllSystems (system: {
        default =
          let
            venv = pythonSets.${system}.mkVirtualEnv "mbf-singlecell-plotter-env" workspace.deps.all;
            pkgs = nixpkgs.legacyPackages.${system};
          in
            venv;
          # pkgs.stdenv.mkDerivation {
          #   name = "mbf-singlecell-plotter-${system}";
          #   buildInputs = [ venv ];
          #   unpackPhase = ":";
          #   installPhase = ''
          #     mkdir -p $out/bin
          #     ln -s ${pythonSets.${system}.python.interpreter} $out/bin/mbf-singlecell-plotter
          #   '';
          # };
      });
    };
}

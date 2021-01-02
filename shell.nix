{ pkgs               ? import <nixpkgs> {
                         overlays = [ ( import ./.nixpkgs/overlay.nix ) ];
                       }
}:

let
  my-python = pkgs.python38.withPackages ( ps: with ps; [
    tensorflow-tensorboard
    pytorch-lightning
    black
    flake8
    matplotlib
    nltk
    numpy
    pandas
    pep8
    pytest
    tqdm
    transformers
    pytorchWithCuda
  ] );

in pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    my-python
  ];
}

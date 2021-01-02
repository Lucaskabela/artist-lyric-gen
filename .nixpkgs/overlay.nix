let
  pythonOverrides = self: super:
    {

      pytorch-lightning = self.callPackage ./pytorch-lightning.nix {
        buildPythonPackage     = super.buildPythonPackage;
        tensorflow-tensorboard = self.tensorflow-tensorboard;
      };

      tensorflow-tensorboard = self.callPackage ./tensorboard.nix {
        buildPythonPackage     = super.buildPythonPackage;
        tensorboard-plugin-wit = self.tensorboard-plugin-wit;
      };

      tensorboard-plugin-wit = self.callPackage ./tensorflow-plugin-wit.nix {
        buildPythonPackage = super.buildPythonPackage;
      };

  };
in self: super: {
  python38 = super.python38.override {
    packageOverrides = pythonOverrides;
  };
}

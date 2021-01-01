self: super:
{

  pytorch-lightning = self.callPackage ./pytorch-lightning.nix {
    buildPythonPackage = super.python38Packages.buildPythonPackage;
  };

  tensorflow-tensorboard = self.callPackage ./tensorboard.nix {
    buildPythonPackage = super.python38Packages.buildPythonPackage;
    tensorboard-plugin-wit = self.tensorboard-plugin-wit;
  };

  tensorboard-plugin-wit = self.callPackage ./tensorflow-plugin-wit.nix {
    buildPythonPackage = super.python38Packages.buildPythonPackage;
  };

}

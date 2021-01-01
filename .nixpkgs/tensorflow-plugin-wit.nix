{ buildPythonPackage, fetchPypi }:

buildPythonPackage rec {
  pname = "tensorboard-plugin-wit";
  version = "1.7.0";
  format = "wheel";
  src = fetchPypi {
    pname = "tensorboard_plugin_wit";
    inherit version format;
    python = "py3";
    sha256 = "ee775f04821185c90d9a0e9c56970ee43d7c41403beb6629385b39517129685b";
  };
}

{ buildPythonPackage, lib, fetchFromGitHub, isPy27, pytestCheckHook
, tensorflow-tensorboard
, future
, pytorch
, pyyaml
, tqdm
, fsspec
}:

buildPythonPackage rec {
  pname = "pytorch-lightning";
  version = "1.1.2";

  disabled = isPy27;

  src = fetchFromGitHub {
    owner = "PyTorchLightning";
    repo = pname;
    rev = version;
    sha256 = "djNH3QSIhR8DXTZqfOUB/K+H9wyQON2DU4b+AMRtDPk=";
  };

  propagatedBuildInputs = [
    tensorflow-tensorboard
    future
    pytorch
    pyyaml
    tqdm
    fsspec
  ];

  checkInputs = [ pytestCheckHook ];
  # Some packages are not in NixPkgs; other tests try to build distributed
  # models, which doesn't work in the sandbox.
  doCheck = false;

  pythonImportsCheck = [ "pytorch_lightning" ];

  meta = with lib; {
    description = "Lightweight PyTorch wrapper for machine learning researchers";
    homepage = "https://pytorch-lightning.readthedocs.io";
    license = licenses.asl20;
    maintainers = with maintainers; [ tbenst ];
  };
}

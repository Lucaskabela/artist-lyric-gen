{ buildPythonPackage, isPy3k, fetchPypi, lib
, numpy
, werkzeug
, protobuf
, markdown
, grpcio
, absl-py
, google-auth-oauthlib
, wheel
, tensorboard-plugin-wit
}:

buildPythonPackage rec {
  pname   = "tensorflow-tensorboard";
  version = "2.2.1";
  format = "wheel";

  disabled = ! isPy3k;

  src = fetchPypi {
    pname = "tensorboard";
    inherit version format;
    python = "py3";
    sha256 = "miotyYVhh2eek/PJXl3Hcd1H4yV9sJdntL4RjXNLTcI=";
  };

  propagatedBuildInputs = [
    numpy
    werkzeug
    protobuf
    markdown
    grpcio
    absl-py
    google-auth-oauthlib
    # not declared in install_requires, but used at runtime
    # https://github.com/NixOS/nixpkgs/issues/73840
    wheel
    tensorboard-plugin-wit
  ];

  # in the absence of a real test suite, run cli and imports
  checkPhase = ''
    $out/bin/tensorboard --help > /dev/null
  '';

  pythonImportsCheck = [
    "tensorboard"
    "tensorboard.backend"
    "tensorboard.compat"
    "tensorboard.data"
    "tensorboard.plugins"
    "tensorboard.summary"
    "tensorboard.util"
  ];

  meta = with lib; {
    description = "TensorFlow's Visualization Toolkit";
    homepage = "http://tensorflow.org";
    license = licenses.asl20;
    maintainers = with maintainers; [ abbradar ];
  };
}

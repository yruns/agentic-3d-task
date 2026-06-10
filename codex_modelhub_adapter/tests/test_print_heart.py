import subprocess
import sys
from pathlib import Path


EXPECTED_OUTPUT = """  **   **
 **** ****
***********
 *********
  *******
   *****
    ***
     *
"""


def test_print_heart_outputs_expected_shape():
    script = Path(__file__).resolve().parents[1] / "print_heart.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == EXPECTED_OUTPUT
    assert result.stderr == ""

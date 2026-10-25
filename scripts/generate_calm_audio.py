import subprocess
from pathlib import Path

out_path = Path("tests/data/speech_sample_calm.wav").resolve()
out_path.parent.mkdir(parents=True, exist_ok=True)

ps_script = f"""
$voice = New-Object -ComObject SAPI.SpVoice
$stream = New-Object -ComObject SAPI.SpFileStream
$stream.Open('{str(out_path)}', 3)
$voice.AudioOutputStream = $stream
$voice.Speak('I feel completely calm, relaxed and peaceful today.')
$stream.Close()
"""

subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True)
print(f"Generated {out_path}, size: {out_path.stat().st_size} bytes")

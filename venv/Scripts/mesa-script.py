#!"C:\Users\ymamo\Google Drive\2. Post PhD\Proxy_Main\ProxyEconomics\venv\Scripts\python.exe"
# EASY-INSTALL-ENTRY-SCRIPT: 'Mesa','console_scripts','mesa'
__requires__ = 'Mesa'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('Mesa', 'console_scripts', 'mesa')()
    )

# STRIX accelerator host probe

Collection is read-only; failures are preserved verbatim.

## uname

+ uname -a
Linux <host> 6.17.0-35-generic #35~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue May 26 19:30:42 UTC 2 x86_64 x86_64 x86_64 GNU/Linux

exit=0

## identity

+ id
uid=1000(<user>) gid=1000(<user>) groups=1000(<user>),65534(nogroup)

exit=0

## groups

+ groups
<user> nogroup

exit=0

## OS release

+ cat /etc/os-release
NAME="Linux Mint"
VERSION="22.3 (Zena)"
ID=linuxmint
ID_LIKE="ubuntu debian"
PRETTY_NAME="Linux Mint 22.3"
VERSION_ID="22.3"
HOME_URL="https://www.linuxmint.com/"
SUPPORT_URL="https://forums.linuxmint.com/"
BUG_REPORT_URL="http://linuxmint-troubleshooting-guide.readthedocs.io/en/latest/"
PRIVACY_POLICY_URL="https://www.linuxmint.com/"
VERSION_CODENAME=zena
UBUNTU_CODENAME=noble

exit=0

## virtualization

+ systemd-detect-virt
none

exit=1

## container virtualization

+ systemd-detect-virt --container
none

exit=1

## PID 1 cgroup

+ cat /proc/1/cgroup
0::/user.slice/user-1000.slice/session-c2.scope

exit=0

## probe-shell cgroup

+ cat /proc/self/cgroup
0::/user.slice/user-1000.slice/session-c2.scope

exit=0

## mount visibility for accelerator paths

+ bash -c grep\ -E\ \'\(/dev\|kfd\|render\|accel\)\'\ /proc/self/mountinfo
819 319 259:2 / / ro,nosuid,nodev,relatime master:1 - ext4 /dev/nvme0n1p2 rw,errors=remount-ro,stripe=64
820 819 0:6 / /dev ro,nosuid,nodev,relatime master:2 - devtmpfs udev rw,size=63654956k,nr_inodes=15913739,mode=755,inode64
821 820 0:25 / /dev/pts ro,nosuid,nodev,noexec,relatime master:3 - devpts devpts rw,gid=5,mode=620,ptmxmode=000
822 820 0:28 / /dev/shm ro,nosuid,nodev master:4 - tmpfs tmpfs rw,inode64
823 820 0:22 / /dev/mqueue ro,nosuid,nodev,noexec,relatime master:15 - mqueue mqueue rw
824 820 0:34 / /dev/hugepages ro,nosuid,nodev,relatime master:16 - hugetlbfs hugetlbfs rw,pagesize=2M
843 819 259:1 / /boot/efi ro,nosuid,nodev,relatime master:30 - vfat /dev/nvme0n1p1 rw,fmask=0077,dmask=0077,codepage=437,iocharset=iso8859-1,shortname=mixed,errors=remount-ro
844 820 0:64 / /dev rw,nosuid,nodev,relatime - tmpfs tmpfs rw,mode=755,uid=1000,gid=1000,inode64
845 844 0:6 /null /dev/null rw,nosuid,relatime master:2 - devtmpfs udev rw,size=63654956k,nr_inodes=15913739,mode=755,inode64
846 844 0:6 /zero /dev/zero rw,nosuid,relatime master:2 - devtmpfs udev rw,size=63654956k,nr_inodes=15913739,mode=755,inode64
847 844 0:6 /full /dev/full rw,nosuid,relatime master:2 - devtmpfs udev rw,size=63654956k,nr_inodes=15913739,mode=755,inode64
848 844 0:6 /random /dev/random rw,nosuid,relatime master:2 - devtmpfs udev rw,size=63654956k,nr_inodes=15913739,mode=755,inode64
849 844 0:6 /urandom /dev/urandom rw,nosuid,relatime master:2 - devtmpfs udev rw,size=63654956k,nr_inodes=15913739,mode=755,inode64
850 844 0:6 /tty /dev/tty rw,nosuid,relatime master:2 - devtmpfs udev rw,size=63654956k,nr_inodes=15913739,mode=755,inode64
851 844 0:65 / /dev/pts rw,nosuid,noexec,relatime - devpts devpts rw,mode=620,ptmxmode=666
852 819 259:2 /tmp /tmp rw,nosuid,nodev,relatime master:1 - ext4 /dev/nvme0n1p2 rw,errors=remount-ro,stripe=64
856 819 259:2 <repo> <repo> rw,nosuid,nodev,relatime master:1 - ext4 /dev/nvme0n1p2 rw,errors=remount-ro,stripe=64
861 852 259:2 /tmp/<sandbox-mount-target> /tmp/<sandbox-mount-target> ro,nosuid,nodev,relatime master:1 - ext4 /dev/nvme0n1p2 rw,errors=remount-ro,stripe=64

exit=0

## namespace links

+ readlink /proc/1/ns/mnt
mnt:[4026533061]

exit=0

## namespace links (self)

+ readlink /proc/self/ns/mnt
mnt:[4026533061]

exit=0

## /dev/kfd

+ ls -l /dev/kfd
ls: cannot access '/dev/kfd': No such file or directory

exit=2

## /dev/dri

+ ls -la /dev/dri
ls: cannot access '/dev/dri': No such file or directory

exit=2

## /dev/accel

+ ls -la /dev/accel
ls: cannot access '/dev/accel': No such file or directory

exit=2

## DRM sysfs classes

+ ls -la /sys/class/drm
total 0
drwxr-xr-x  2 nobody nogroup    0 Aug 24 17:17 .
drwxr-xr-x 79 nobody nogroup    0 Aug 24 17:17 ..
lrwxrwxrwx  1 nobody nogroup    0 Aug 18 07:05 card0 -> ../../devices/pci0000:00/0000:00:08.1/0000:c5:00.0/drm/card0
lrwxrwxrwx  1 nobody nogroup    0 Aug 20 12:17 card0-DP-1 -> ../../devices/pci0000:00/0000:00:08.1/0000:c5:00.0/drm/card0/card0-DP-1
lrwxrwxrwx  1 nobody nogroup    0 Aug 20 12:17 card0-DP-2 -> ../../devices/pci0000:00/0000:00:08.1/0000:c5:00.0/drm/card0/card0-DP-2
lrwxrwxrwx  1 nobody nogroup    0 Aug 20 12:17 card0-DP-3 -> ../../devices/pci0000:00/0000:00:08.1/0000:c5:00.0/drm/card0/card0-DP-3
lrwxrwxrwx  1 nobody nogroup    0 Aug 20 12:17 card0-DP-4 -> ../../devices/pci0000:00/0000:00:08.1/0000:c5:00.0/drm/card0/card0-DP-4
lrwxrwxrwx  1 nobody nogroup    0 Aug 20 12:17 card0-DP-5 -> ../../devices/pci0000:00/0000:00:08.1/0000:c5:00.0/drm/card0/card0-DP-5
lrwxrwxrwx  1 nobody nogroup    0 Aug 20 12:17 card0-DP-6 -> ../../devices/pci0000:00/0000:00:08.1/0000:c5:00.0/drm/card0/card0-DP-6
lrwxrwxrwx  1 nobody nogroup    0 Aug 20 12:17 card0-DP-7 -> ../../devices/pci0000:00/0000:00:08.1/0000:c5:00.0/drm/card0/card0-DP-7
lrwxrwxrwx  1 nobody nogroup    0 Aug 20 12:17 card0-DP-8 -> ../../devices/pci0000:00/0000:00:08.1/0000:c5:00.0/drm/card0/card0-DP-8
lrwxrwxrwx  1 nobody nogroup    0 Aug 20 12:17 card0-HDMI-A-1 -> ../../devices/pci0000:00/0000:00:08.1/0000:c5:00.0/drm/card0/card0-HDMI-A-1
lrwxrwxrwx  1 nobody nogroup    0 Aug 20 12:17 card0-Writeback-1 -> ../../devices/pci0000:00/0000:00:08.1/0000:c5:00.0/drm/card0/card0-Writeback-1
lrwxrwxrwx  1 nobody nogroup    0 Aug 19 20:57 renderD128 -> ../../devices/pci0000:00/0000:00:08.1/0000:c5:00.0/drm/renderD128
-r--r--r--  1 nobody nogroup 4096 Aug 24 17:17 version

exit=0

## accelerator sysfs classes

+ ls -la /sys/class/accel
total 0
drwxr-xr-x  2 nobody nogroup 0 Aug 24 17:17 .
drwxr-xr-x 79 nobody nogroup 0 Aug 24 17:17 ..
lrwxrwxrwx  1 nobody nogroup 0 Aug 19 20:57 accel0 -> ../../devices/pci0000:00/0000:00:08.2/0000:c6:00.1/accel/accel0

exit=0

## device ownership groups

+ getent group video
video:x:44:<user>,ollama,friend

exit=0

## device ownership groups

+ getent group render
render:x:992:<user>,ollama,lemonade,friend

exit=0

## device ownership groups

+ getent group input
input:x:995:

exit=0

## PCI devices and drivers

+ lspci -nnk
pcilib: Error reading /sys/bus/pci/devices/0000:00:08.3/label: Operation not permitted
00:00.0 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Root Complex [1022:1507] (rev 02)
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Root Complex [1022:1507]
00:00.2 IOMMU [0806]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo IOMMU [1022:1508] (rev 02)
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo IOMMU [1022:1508]
00:01.0 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Dummy Host Bridge [1022:1509]
00:01.1 PCI bridge [0604]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo PCIe USB4 Bridge [1022:150a] (rev 02)
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo PCIe USB4 Bridge [1022:150a]
	Kernel driver in use: pcieport
00:01.2 PCI bridge [0604]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo PCIe USB4 Bridge [1022:150a] (rev 02)
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo PCIe USB4 Bridge [1022:150a]
	Kernel driver in use: pcieport
00:02.0 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Dummy Host Bridge [1022:1509]
00:02.1 PCI bridge [0604]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo GPP Bridge [1022:150b] (rev 02)
	Subsystem: Advanced Micro Devices, Inc. [AMD] Device [1022:1453]
	Kernel driver in use: pcieport
00:02.2 PCI bridge [0604]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo GPP Bridge [1022:150b] (rev 02)
	Subsystem: Advanced Micro Devices, Inc. [AMD] Device [1022:1453]
	Kernel driver in use: pcieport
00:02.3 PCI bridge [0604]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo GPP Bridge [1022:150b] (rev 02)
	Subsystem: Advanced Micro Devices, Inc. [AMD] Device [1022:1453]
	Kernel driver in use: pcieport
00:03.0 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Dummy Host Bridge [1022:1509]
00:03.1 PCI bridge [0604]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo GPP Bridge [1022:150b]
	Subsystem: Advanced Micro Devices, Inc. [AMD] Device [1022:1453]
	Kernel driver in use: pcieport
00:08.0 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Dummy Host Bridge [1022:1509]
00:08.1 PCI bridge [0604]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Internal GPP Bridge to Bus [C:A] [1022:150c]
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Internal GPP Bridge to Bus [C:A] [1022:150c]
	Kernel driver in use: pcieport
00:08.2 PCI bridge [0604]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Internal GPP Bridge to Bus [C:A] [1022:150c]
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Internal GPP Bridge to Bus [C:A] [1022:150c]
	Kernel driver in use: pcieport
00:08.3 PCI bridge [0604]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Internal GPP Bridge to Bus [C:A] [1022:150c]
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Internal GPP Bridge to Bus [C:A] [1022:150c]
	Kernel driver in use: pcieport
00:14.0 SMBus [0c05]: Advanced Micro Devices, Inc. [AMD] FCH SMBus Controller [1022:790b] (rev 71)
	Subsystem: Advanced Micro Devices, Inc. [AMD] FCH SMBus Controller [1022:790b]
	Kernel driver in use: piix4_smbus
	Kernel modules: i2c_piix4, sp5100_tco
00:14.3 ISA bridge [0601]: Advanced Micro Devices, Inc. [AMD] FCH LPC Bridge [1022:790e] (rev 51)
	Subsystem: Advanced Micro Devices, Inc. [AMD] FCH LPC Bridge [1022:790e]
00:18.0 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix Halo Data Fabric; Function 0 [1022:12b8]
00:18.1 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix Halo Data Fabric; Function 1 [1022:12b9]
00:18.2 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix Halo Data Fabric; Function 2 [1022:12ba]
00:18.3 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix Halo Data Fabric; Function 3 [1022:12bb]
	Kernel driver in use: k10temp
	Kernel modules: k10temp
00:18.4 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix Halo Data Fabric; Function 4 [1022:12bc]
00:18.5 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix Halo Data Fabric; Function 5 [1022:12bd]
00:18.6 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix Halo Data Fabric; Function 6 [1022:12be]
00:18.7 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix Halo Data Fabric; Function 7 [1022:12bf]
c1:00.0 Ethernet controller [0200]: Realtek Semiconductor Co., Ltd. RTL8125 2.5GbE Controller [10ec:8125] (rev 05)
	DeviceName: OnBoard LAN
	Subsystem: Realtek Semiconductor Co., Ltd. RTL8125 2.5GbE Controller [10ec:8125]
	Kernel driver in use: r8169
	Kernel modules: r8169
c2:00.0 SD Host controller [0805]: Genesys Logic, Inc GL9755 SD Host Controller [17a0:9755] (rev 01)
	Subsystem: Genesys Logic, Inc GL9755 SD Host Controller [17a0:9755]
	Kernel driver in use: sdhci-pci
	Kernel modules: sdhci_pci
c3:00.0 Network controller [0280]: MEDIATEK Corp. MT7925 (RZ717) Wi-Fi 7 160MHz [14c3:0717]
	Subsystem: MEDIATEK Corp. MT7925 (RZ717) Wi-Fi 7 160MHz [14c3:0717]
	Kernel driver in use: mt7925e
	Kernel modules: mt7925e
c4:00.0 Non-Volatile memory controller [0108]: Sandisk Corp WD_BLACK SN7100/WD PC SN7100S M.2 2280 NVMe SSD (DRAM-less) [15b7:5045] (rev 01)
	Subsystem: Sandisk Corp WD_BLACK SN7100/WD PC SN7100S M.2 2280 NVMe SSD (DRAM-less) [15b7:5045]
	Kernel driver in use: nvme
	Kernel modules: nvme
c5:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] Strix Halo [Radeon Graphics / Radeon 8050S Graphics / Radeon 8060S Graphics] [1002:1586] (rev c1)
	Subsystem: Device [2014:801d]
	Kernel driver in use: amdgpu
	Kernel modules: amdgpu
c5:00.1 Audio device [0403]: Advanced Micro Devices, Inc. [AMD/ATI] Radeon High Definition Audio Controller [1002:1640]
	Subsystem: Device [2014:801d]
	Kernel driver in use: snd_hda_intel
	Kernel modules: snd_hda_intel
c5:00.2 Encryption controller [1080]: Advanced Micro Devices, Inc. [AMD] Strix/Krackan/Strix Halo CCP/ASP [1022:17e0]
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix/Krackan/Strix Halo CCP/ASP [1022:17e0]
	Kernel driver in use: ccp
	Kernel modules: ccp
c5:00.4 USB controller [0c03]: Advanced Micro Devices, Inc. [AMD] Strix Halo USB 3.1 xHCI [1022:1587]
	Subsystem: Advanced Micro Devices, Inc. [AMD] Device [1022:15b9]
	Kernel driver in use: xhci_hcd
c5:00.6 Audio device [0403]: Advanced Micro Devices, Inc. [AMD] Ryzen HD Audio Controller [1022:15e3]
	DeviceName: OnBoard Audio
	Subsystem: Device [2014:801d]
	Kernel driver in use: snd_hda_intel
	Kernel modules: snd_hda_intel
c6:00.0 Non-Essential Instrumentation [1300]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo PCIe Dummy Function [1022:150d]
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo PCIe Dummy Function [1022:150d]
c6:00.1 Signal processing controller [1180]: Advanced Micro Devices, Inc. [AMD] Strix/Krackan/Strix Halo Neural Processing Unit [1022:17f0] (rev 11)
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix/Krackan/Strix Halo Neural Processing Unit [1022:17f0]
	Kernel driver in use: amdxdna
	Kernel modules: amdxdna
c7:00.0 USB controller [0c03]: Advanced Micro Devices, Inc. [AMD] Strix Halo USB 3.1 xHCI [1022:1588]
	Subsystem: Advanced Micro Devices, Inc. [AMD] Device [1022:15b9]
	Kernel driver in use: xhci_hcd
c7:00.3 USB controller [0c03]: Advanced Micro Devices, Inc. [AMD] Strix Halo USB 3.1 xHCI [1022:1589]
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix Halo USB 3.1 xHCI [1022:1589]
	Kernel driver in use: xhci_hcd
c7:00.4 USB controller [0c03]: Advanced Micro Devices, Inc. [AMD] Strix Halo USB 3.1 xHCI [1022:158b]
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix Halo USB 3.1 xHCI [1022:158b]
	Kernel driver in use: xhci_hcd
c7:00.5 USB controller [0c03]: Advanced Micro Devices, Inc. [AMD] Strix Halo USB4 Host Router [1022:158d]
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix Halo USB4 Host Router [1022:158d]
	Kernel driver in use: thunderbolt
	Kernel modules: thunderbolt
c7:00.6 USB controller [0c03]: Advanced Micro Devices, Inc. [AMD] Strix Halo USB4 Host Router [1022:158e]
	Subsystem: Advanced Micro Devices, Inc. [AMD] Strix Halo USB4 Host Router [1022:158e]
	Kernel driver in use: thunderbolt
	Kernel modules: thunderbolt

exit=0

## loaded AMD modules

+ bash -c lsmod\ \|\ grep\ -E\ \'\(amdgpu\|amdxdna\|xrt\)\'
amdgpu              20107264  3
amdxcp                 12288  1 amdgpu
drm_panel_backlight_quirks    12288  1 amdgpu
drm_buddy              28672  1 amdgpu
drm_ttm_helper         16384  1 amdgpu
ttm                   126976  2 amdgpu,drm_ttm_helper
drm_exec               12288  1 amdgpu
drm_suballoc_helper    24576  1 amdgpu
drm_display_helper    290816  1 amdgpu
cec                    98304  2 drm_display_helper,amdgpu
i2c_algo_bit           16384  1 amdgpu
amdxdna               159744  1
gpu_sched              65536  2 amdxdna,amdgpu
video                  77824  1 amdgpu

exit=0

## amdxdna module metadata

+ modinfo amdxdna
filename:       /lib/modules/6.17.0-35-generic/updates/dkms/amdxdna.ko.zst
import_ns:      DMA_BUF
description:    amdxdna driver
author:         XRT Team <runtimeca39d@amd.com>
license:        GPL
firmware:       amdnpu/17f0_11/npu_7.sbin
firmware:       amdnpu/17f0_10/npu_7.sbin
firmware:       amdnpu/1502_00/npu_7.sbin
firmware:       amdnpu/17f0_20/npu.sbin
firmware:       amdnpu/17f0_11/npu.sbin
firmware:       amdnpu/17f0_10/npu.sbin
firmware:       amdnpu/1502_00/npu.sbin
srcversion:     6897770A6CF79C855D3FA2E
alias:          pci:v00001022d000017F0sv*sd*bc*sc*i*
alias:          pci:v00001022d00001502sv*sd*bc*sc*i*
depends:        gpu-sched
name:           amdxdna
retpoline:      Y
vermagic:       6.17.0-35-generic SMP preempt mod_unload modversions
sig_id:         PKCS#7
signer:         localhost.localdomain Secure Boot Module Signature key
sig_key:        86:D7:FF:CE:16:8B:01:3C:0E:13:F5:61:D8:71:75:90:27:71:ED
sig_hashalgo:   sha512
signature:      1D:A1:F8:C5:3F:5D:60:34:26:8C:55:75:07:CD:61:BA:EF:03:71:78:
		20:74:B1:BD:DC:5C:DA:B7:B0:40:0D:0B:BA:6E:74:A0:33:BA:03:AC:
		D2:4F:E5:66:0E:9D:1A:DC:00:B7:0E:A5:E2:B4:94:58:43:E8:51:F6:
		27:C6:4F:1B:37:D0:03:2D:56:F0:D2:76:28:69:AB:1F:77:B6:7A:56:
		1D:F7:84:43:D8:88:0C:93:9A:83:35:66:7A:BD:7B:E9:74:4A:BF:59:
		33:40:BA:75:A0:99:67:A1:2A:D7:07:4B:EC:DA:1C:55:DE:D6:2F:61:
		80:61:D8:5C:67:7E:2B:12:57:8C:21:FD:9F:22:36:DF:ED:9A:4C:D9:
		49:C2:15:4A:EC:80:E9:1D:4C:DE:32:27:A5:ED:C5:07:22:B4:DD:0C:
		45:03:ED:61:59:F3:4A:3E:83:0C:39:61:60:84:87:6F:DF:1D:B5:9F:
		BB:3A:E0:26:A6:BC:C1:B7:63:18:98:21:91:E0:2B:3E:F1:E0:53:B4:
		F0:06:F0:DC:C6:75:57:4B:B9:B5:3A:6F:56:FA:B8:D6:87:87:C3:D3:
		99:A8:83:AA:FC:3E:F2:14:91:BE:B5:41:E1:1B:7A:3E:44:03:88:E6:
		09:53:F9:79:00:50:91:5C:E3:E7:E0:4E:83:D7:D0:B8
parm:           aie2_max_col:Maximum column could be used (uint)
parm:           force_cmdlist:Force use command list (Default true) (bool)

exit=0

## installed accelerator packages

+ dpkg-query -W -f=\$\{Package\}\ \$\{Version\}\\n amdxdna-dkms libxrt-npu2 libxrt-utils-npu libxrt2 fastflowlm
amdxdna-dkms 7.0.0-rc1+git20260310.6b13cb8f4-noble1
fastflowlm 0.9.39
libxrt-npu2 1:2.21.75-1~noble1
libxrt-utils-npu 1:2.21.75-1~noble1
libxrt2 1:2.21.75-1~noble1

exit=0

## rocminfo

+ rocminfo
[37mROCk module is loaded[0m
[31mUnable to open /dev/kfd read-write: No such file or directory[0m
[31mFailed to get user name to check for video group membership[0m

exit=1

## rocm-smi

+ rocm-smi


======================================== ROCm System Management Interface ========================================
================================================== Concise Info ==================================================
Device  Node  IDs              Temp    Power     Partitions          SCLK  MCLK  Fan  Perf  PwrCap  VRAM%  GPU%
[3m              (DID,     GUID)  (Edge)  (Socket)  (Mem, Compute, ID)                                              [0m
==================================================================================================================
0       1     0x1586,   51259  39.0°C  13.057W   N/A, N/A, 0         N/A   N/A   0%   auto  N/A     3%     0%
==================================================================================================================
============================================== End of ROCm SMI Log ===============================================

exit=0

## amd-smi list

+ amd-smi list
GPU: 0
    BDF: 0000:c5:00.0
    UUID: 00ff1586-0000-1000-8000-000000000000
    KFD_ID: 51259
    NODE_ID: 1
    PARTITION_ID: 0


exit=0

## xrt-smi version

+ xrt-smi --version
  Version              : 2.21.75
  virtio-pci Version   : 6.17.0-35-generic
  amdxdna Version      : 6.17.0-35-generic

exit=0

## xrt-smi examine

+ xrt-smi examine
System Configuration
  OS Name              : Linux
  Release              : 6.17.0-35-generic
  Machine              : x86_64
  CPU Cores            : 32
  Memory               : 124415 MB
  Distribution         : Linux Mint 22.3
  GLIBC                : 2.39
  Model                : MME3L
  BIOS Vendor          : American Megatrends International, LLC.
  BIOS Version         : 3.05
  Processor            : AMD RYZEN AI MAX+ 395 w/ Radeon 8060S

XRT
  Version              : 2.21.75
  virtio-pci Version   : 6.17.0-35-generic
  amdxdna Version      : 6.17.0-35-generic

Device(s) Present
  0 devices found

exit=0

## flm version

+ flm --version
FLM v0.9.39

exit=1

## flm validate

+ flm validate
[Linux]  Kernel: 6.17.0-35-generic
[31m[ERROR]  No NPU device found.[0m
[32m[Linux]  Memlock Limit: infinity[0m

exit=1

## flm validate JSON

+ flm validate --json
{
    "all_fw_ok": true,
    "amd_device_found": false,
    "devices": [],
    "enough_cols": true,
    "kernel": "6.17.0-35-generic",
    "kernel_ok": true,
    "memlock_limit": "infinity",
    "memlock_ok": true,
    "object": "npu_stack_validation",
    "platform": "linux",
    "ready": false
}

exit=1

## xdna-top snapshot

+ xdna-top --json snapshot
{
  "backends": {
    "igpu": {
      "primary": "sysfs",
      "signals": {
        "busy_pct": "sysfs",
        "power_w": "sysfs"
      }
    },
    "npu": {
      "fallbacks_used": [],
      "primary": "xrt_smi",
      "signals": {
        "contexts": null,
        "device": null,
        "driver": null,
        "power_state": null,
        "sensors": null
      }
    }
  },
  "captured_at": "2026-08-25T00:47:57.640141Z",
  "commands": {
    "xrt_smi": {
      "aie_partitions_returncode": 1,
      "available": true,
      "examine_returncode": 0,
      "path": "/usr/bin/xrt-smi",
      "version_output": "Version : 2.21.75 virtio-pci Version : 6.17.0-35-generic amdxdna Version : 6.17.0-35-generic"
    }
  },
  "degraded": {
    "igpu": {
      "degraded": false,
      "reasons": []
    },
    "npu": {
      "degraded": true,
      "reasons": [
        "aie_partitions_report_failed"
      ]
    },
    "overall": true
  },
  "devices": {
    "accel": {
      "entries": [
        {
          "exists": false,
          "path": "/dev/accel/accel0"
        }
      ]
    },
    "igpu": {
      "busy_path": "/sys/class/drm/card0/device/gpu_busy_percent",
      "busy_path_exists": true,
      "power_path": "/sys/class/hwmon/hwmon5/power1_input",
      "power_path_exists": true
    },
    "npu": {
      "bdf": null,
      "contexts": [],
      "detected": false,
      "driver": {
        "drm_version": null,
        "supports_sensors": false
      },
      "ioctl": {
        "aie": null,
        "aie_version": null,
        "available": false,
        "clocks": null,
        "driver": null,
        "firmware_version": null,
        "node": null,
        "reason": "accel_absent",
        "sensors": {
          "available": false,
          "items": [],
          "reason": null
        },
        "source": null,
        "supports_sensors": false
      },
      "name": null,
      "power_state": {
        "available": false,
        "dpm": null,
        "powerstate": null,
        "reason": "debugfs_accel_absent",
        "source": null
      },
      "report_shape": {
        "context_count": 0,
        "has_aie_partitions": false,
        "has_hw_contexts": false
      },
      "sensors": {
        "column_utilization_pct": {
          "source": null,
          "value": null
        },
        "power_w": {
          "source": null,
          "value": null
        }
      }
    }
  },
  "errors": [
    {
      "message": "ERROR: Please specify a device using --device option Available devices:",
      "probe": "xrt_smi.aie_partitions"
    }
  ],
  "host": {
    "hostname": "<host>",
    "kernel": {
      "release": "6.17.0-35-generic",
      "version": "#35~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue May 26 19:30:42 UTC 2"
    },
    "os": {
      "id": "linuxmint",
      "pretty_name": "Linux Mint 22.3",
      "version_id": "22.3"
    },
    "python": {
      "version": "3.12.3"
    },
    "xdna_top": {
      "version": "0.1.0"
    }
  },
  "kind": "xdna-top.snapshot",
  "schema_version": "1.0",
  "telemetry": {
    "gpu_busy_pct": 0,
    "gpu_power_w": 14.057,
    "igpu_degraded": false,
    "npu_active": false,
    "npu_degraded": true,
    "state": "IDLE",
    "ts": 1787618877.6399262
  }
}

exit=0

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# -----------------------------
# Funciones teóricas
# -----------------------------
def ber_qpsk(EbN0_dB):
    EbN0 = 10**(EbN0_dB/10)
    return 0.5 * erfc(np.sqrt(EbN0))

def ber_mqam(EbN0_dB, M):
    k = np.log2(M)
    EbN0 = 10**(EbN0_dB/10)
    return (4/k)*(1 - 1/np.sqrt(M)) * 0.5 * erfc(np.sqrt(3*k*EbN0/(2*(M-1))))

# -----------------------------
# Main
# -----------------------------
EbN0_dB = np.arange(0, 21, 0.5)

# Teórico
ber_qpsk_th = ber_qpsk(EbN0_dB)
ber_16qam_th = ber_mqam(EbN0_dB, 16)
ber_64qam_th = ber_mqam(EbN0_dB, 64)

# -----------------------------
# Plot
# -----------------------------
plt.figure()

plt.semilogy(EbN0_dB, ber_qpsk_th, '--', label='QPSK (teo)')
plt.semilogy(EbN0_dB, ber_16qam_th, '--', label='16-QAM (teo)')
plt.semilogy(EbN0_dB, ber_64qam_th, '--', label='64-QAM (teo)')

plt.ylim([1e-6, 5e-2])
plt.grid(True, which='both')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.title('BER vs Eb/N0')
plt.legend()

plt.show()
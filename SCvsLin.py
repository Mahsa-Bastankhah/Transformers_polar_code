import itertools
import numpy as np
from polar import *

parser = argparse.ArgumentParser(description='Polar/PAC code - decoder')
args = parser.parse_args()
args.hard_decision = False
# Parameters
N = 16  # Codeword length
K = 8   # Message length
SNR_dB = 3  # Signal-to-noise ratio in dB
info_inds = [7,9,10,11,12,13,14,15]


# Generate all possible 8-bit messages
alphabet = np.array([-1, 1])
msg_bits = np.array(list(itertools.product(alphabet, repeat=K)))
msg_bits = torch.tensor(msg_bits, dtype=torch.float32)
gt = torch.ones(len(msg_bits), N)
gt[:, info_inds] = msg_bits


n = int(np.log2(N))
rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1
# Multiple SNRs:
polar = PolarCode(n, K, args, rs=rs)


# Encode message to get the codeword
polar_code = polar.encode_plotkin(msg_bits,custom_info_positions = info_inds)
noisy_code = polar.channel(polar_code, SNR_dB)
SC_llrs, decoded_SC_msg_bits = polar.sc_decode_new(noisy_code, SNR_dB )

# Convert decoded messages to numpy array for comparison
decoded_bits_np = decoded_SC_msg_bits.numpy()

# Calculate Bitwise Error Rate (BER) for each bit position
bitwise_errors = np.sum(decoded_bits_np != msg_bits.numpy(), axis=0)
total_bits_per_bit = gt.size(0)  # Assuming gt is a 2D tensor

# Calculate Bitwise Error Rate (BER) for each bit position
ber_per_bit = bitwise_errors / total_bits_per_bit

# Print the result
for bit_position, ber in enumerate(ber_per_bit):
    print(f"Bitwise Error Rate (BER) for Bit {bit_position}: {ber}")

print("Mean BER of SC: ", np.mean(ber_per_bit , axis=0))





from typing import Any, Dict, Iterable, Tuple


def parse_kernel_entry(entry: Iterable) -> Tuple[str, Any, Any, str]:
	"""Parse a kernel `entry` tuple produced by BaseKernelInfoParser.individual_info.

	The original test previously unpacked entries with the following pattern:
		(kernel_name, input_dim, dtype, _, _, _, op, _, _, _) = entry

	This helper returns a small normalized tuple:
		(kernel_name, input_dim, dtype, op)

	It is defensive: if the entry is shorter than expected or values are 'N/A',
	it will normalize them to None where appropriate.
	"""

	# defensively coerce to list for index access
	e = list(entry)
	# extract by index with fallback
	kernel_name = e[0] if len(e) > 0 else None
	input_dim = e[1] if len(e) > 1 else None
	dtype = e[2] if len(e) > 2 else None
	op = e[6] if len(e) > 6 else None

	# normalize common 'N/A' sentinel
	if input_dim == "N/A" or input_dim == []:
		input_dim = None
	if dtype == "N/A":
		dtype = None
	if op == "N/A":
		op = None

	return kernel_name, input_dim, dtype, op


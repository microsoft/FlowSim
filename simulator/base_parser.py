import gzip
import json
import csv
import os

from collections import defaultdict
import re


class BaseKernelInfoParser:
    """
    Parses kernel trace files and extracts detailed information for each kernel event.

    This class processes profiling trace files to retrieve kernel execution details,
    including names, input dimensions, data types, durations, and call stack info.
    It supports mapping external identifiers to kernel metadata and aggregates kernel
    statistics for further analysis.

    Args:
        file_path (str): Path to the gzipped JSON trace file containing event data.

    Attributes:
        file_path (str): The input trace file path.
        base_filename (str): The base name of the input file.
        base_name (str): The base name without extensions.
        events (list): List of parsed trace events.
        individual_info (list): List of tuples with kernel event details.
        total_duration (int): Total duration of all kernel events.

    Returns:
        individual_info (list): A list of tuples, each containing:
            - name (str): Kernel name.
            - dims (any): Input dimensions.
            - input_type (any): Data type of input.
            - roles (str): Placeholder for input/output roles.
            - desc (str): Placeholder for descriptions.
            - duration (int): Duration of the kernel event (us).
            - op (str): Placeholder for operation.
            - operation (str): Placeholder for operation details.
            - source_code (str): Placeholder for source code reference.
            - call_stack (str): Human-readable call stack trace.

    Example:
        [
            ('aten::matmul', [[64, 128], [128, 256]], ['float32', 'float32'],
             '', '', 123, '', '', '', 'LaunchKernel <- forward <- Main'),
            ...
        ]
    """

    def __init__(self, file_path: str):
        """Initializes the BaseKernelInfoParser object.

        Loads and parses kernel trace events from the specified gzipped JSON file,
        preparing internal structures for kernel profiling analysis.

        Args:
            file_path (str): Path to the gzipped JSON trace file containing kernel events.

        Attributes:
            file_path (str): Stores the input file path.
            base_filename (str): The base name of the input file.
            base_name (str): The base name without extensions.
            events (list): List of parsed trace events.
            individual_info (list): List of tuples containing kernel event details.
            total_duration (int): Total duration of all kernel events.

        Returns:
            None. Initializes internal state and populates event data for further
            analysis.
        """
        self.file_path = file_path
        self.base_filename = os.path.basename(file_path)
        self.base_name = os.path.splitext(
            os.path.splitext(self.base_filename)[0]
        )[0]

        self.events = []
        # individual_info = [(name, dims, input_type, roles, desc, duration, op, operation, source_code, call_stack)]
        self.individual_info = []
        self.aggregate_kernel_info = []
        self.total_duration = 0

        self._load_events()
        self._parse_events()

    def _load_events(self) -> None:
        """
        Loads and parses kernel trace events from a gzipped JSON file.

        Args:
            None. Uses self.file_path as the source file path.

        Returns:
            None. Populates self.events with a list of parsed event dictionaries.
            If loading fails, self.events will be an empty list.
        """
        try:
            with gzip.open(self.file_path, "rt", encoding="utf-8") as f:
                full_data = json.load(f)
            self.events = full_data.get("traceEvents", [])
        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")
            self.events = []

    def _get_callstack(self) -> dict[int, str]:
        """
        Builds a mapping from external or correlation IDs to call stack traces
        for kernel launch events.

        Processes the trace events in self.events, grouping them by process and
        thread IDs, and simulates call stack nesting using event timestamps and
        durations. For each event with an "External id" or "correlation"
        argument and a call stack containing "LaunchKernel", associates the ID
        with the corresponding call stack trace.

        Args:
            None. Uses self.events (list of event dicts) as input.

        Returns:
            A dict mapping external IDs or correlation IDs (as found in event
            arguments) to a string representing the call stack trace at the time
            of kernel launch. The call stack trace is formatted as
            "stack top <- ... <- stack base".

            Example:
            {
                12345: "LaunchKernel <- forward <- Main"
            }
        """
        threads = defaultdict(list)
        extid_to_stack = {}
        for evt in self.events:
            # The complete event is marked with "ph": "X"
            # Incomplete events (like "B" and "E") are not happening in the trace
            if evt.get("ph") == "X":
                pid = evt.get("pid")
                tid = evt.get("tid")
                threads[(pid, tid)].append(evt)

        for (pid, tid), evts in threads.items():
            # Sort events by timestamp for proper nesting
            evts.sort(key=lambda e: e["ts"])

            call_stack = []
            for evt in evts:
                start = evt["ts"]
                dur = evt.get("dur", 0)
                end = start + dur
                name = evt["name"]
                # By design, external_id and correlation_id are mutually exclusive
                external_id = evt.get("args", {}).get("External id")
                correlation_id = evt.get("args", {}).get("correlation")

                # Pop finished frames: ensure the current call_stack only contains
                # events that are still active at 'start'. Each stack entry stores
                # (name, end_time, external_id, correlation_id). If the top frame's
                # end time is <= current event's start, that frame finished before
                # this event began, so remove it.
                while call_stack and call_stack[-1][1] <= start:
                    call_stack.pop()

                # Push current event onto the simulated call stack. We store end time
                # and any ids so we can later map kernels back to annotations.
                call_stack.append((name, end, external_id, correlation_id))

                # Build human readable call stack string from outermost to innermost
                stack_trace = " <- ".join(
                    name for name, _, _, _ in reversed(call_stack)
                )

                # Only record mappings for events that have an External id or a
                # correlation id and where the stack contains a LaunchKernel entry.
                # Reason: we want to associate user-facing ids (External id / correlation)
                # with the higher-level call context that launched GPU kernels.
                if (
                    external_id is not None or correlation_id is not None
                ) and "LaunchKernel" in stack_trace:
                    if external_id is not None:
                        extid_to_stack[external_id] = stack_trace
                    elif correlation_id is not None:
                        extid_to_stack[correlation_id] = stack_trace
                    else:
                        # This should never happen, otherwise the kernel is obsolete
                        print(
                            f"Error: Event {name} at {start}us has no External id or correlation id."
                        )

        return extid_to_stack

    def _parse_events(self) -> list[tuple]:
        """
        Parses kernel trace events and extracts structured kernel execution details.

        Processes self.events, a list of event dictionaries from a profiler trace,
        to build a summary of GPU kernel executions. For each kernel event, this
        method maps external IDs to input dimensions and types, matches kernel
        events with user/system annotations, and retrieves the call stack for each
        kernel. The extracted information is stored in self.individual_info as a
        list of tuples, each containing:

            (name, dims, input_type, roles, desc, duration, op, operation,
             source_code, call_stack)

        Args:
            None. Operates on self.events, which should be populated with profiling
            event data.

        Returns:
            list: A list of tuples, each representing a kernel event with the
            following fields:
            - name (str): Kernel name.
            - dims (any): Input dimensions.
            - input_type (any): Data type of input.
            - roles (str): Placeholder for input/output roles.
            - desc (str): Placeholder for descriptions.
            - duration (int): Duration of the kernel event in microseconds.
            - op (str): Placeholder for operation.
            - operation (str): Placeholder for operation details.
            - source_code (str): Placeholder for source code reference.
            - call_stack (str): Human-readable call stack trace.

        Example:
            [
            ('aten::matmul', [[64, 128], [128, 256]],
             ['float32', 'float32'], '', '', 123, '', '', '',
             'LaunchKernel <- forward <- Main'),
            ...
            ]
        """
        extid_to_dims = {}
        extid_to_type = {}
        self.individual_info = []
        annotation_events = [
            entry
            for entry in self.events
            if (
                entry.get("cat") == "gpu_user_annotation"
                or entry.get("cat") == "user_annotation"
            )
        ]

        # Map External id to Input Dimensions and Input type
        for entry in self.events:
            args = entry.get("args", {})
            ext_id = args.get("External id")
            dims = args.get("Input Dims")
            input_type = args.get("Input type")
            if ext_id is not None and dims is not None:
                extid_to_dims[ext_id] = dims
                extid_to_type[ext_id] = input_type

        # Get call stack for each kernel using External id or correlation id
        kernel_call_stack = self._get_callstack()
        query_name_counter = defaultdict(int)

        for entry in self.events:
            if entry.get("cat") == "kernel":
                # Kernel event intermediate parameters
                args = entry.get("args", {})
                ext_id = args.get("External id")
                correlation_id = args.get("correlation")
                start = entry.get("ts", 0)
                end = start + entry.get("dur", 0)
                # Kernel information to be recorded
                # Format: [(name, dims, input_type, roles, desc, duration, op, operation, source_code, call_stack)]
                name = entry.get("name")
                dims = extid_to_dims.get(ext_id, "N/A")
                input_type = extid_to_type.get(ext_id, "N/A")
                duration = entry.get("dur", 0)
                # roles, desc, op, operation, source_code are added from database later
                idx = query_name_counter[name]
                query_name_counter[name] += 1
                if ext_id is not None:
                    call_stack = kernel_call_stack.get(ext_id)
                elif correlation_id is not None:
                    call_stack = kernel_call_stack.get(correlation_id)
                else:
                    call_stack = None
                    print(
                        f"Warning: kernel {name} at {start}us has no upper stream External id or correlation id."
                    )

                # Case 1: Torch profiler linked external id with input dims and type
                if ext_id is not None and dims != "N/A":
                    self.individual_info.append(
                        (
                            name,
                            dims,
                            input_type,
                            "",
                            "",
                            duration,
                            "",
                            "",
                            "",
                            call_stack,
                        )
                    )
                else:
                    # Case 2: If no ext_id, we need to find the shape from user annotations
                    # Key Identification Methodology: Annotation is overlapped with kernel
                    dims_anno = "N/A"
                    input_type_anno = "N/A"
                    for anno in annotation_events:
                        if "ProfilerStep" in anno.get("name", ""):
                            continue
                        anno_start = anno.get("ts", 0)
                        anno_end = anno_start + anno.get("dur", 0)
                        if "nccl" in name.lower():
                            buffer = 1000  # 1ms buffer for NCCL annotations due to launch delay
                        else:
                            buffer = 1  # 1us buffer for almost overlapping annotations
                        # Check if the kernel's time range overlaps with the annotation's time range
                        if anno_start - buffer <= start <= anno_end + buffer:
                            if "nccl" in name.lower():
                                # Avoid nccl kernel matching other annotations
                                if "nccl" in anno.get(
                                    "name", ""
                                ) or "attn_tp_reduce_scatter" in anno.get(
                                    "name", ""
                                ):
                                    # Annotation Style 1: User Injected, information included in name
                                    name_anno = anno.get("name")
                                    dims_anno = re.findall(
                                        r"(\w+=\([^)]+\))", name_anno
                                    )
                                    dims_anno = [
                                        list(map(int, re.findall(r"\d+", s)))
                                        for s in dims_anno
                                    ]

                                    input_type_anno_match = re.search(
                                        r"dtype=([^\]]+)\]", name_anno
                                    )
                                    input_type_anno = (
                                        input_type_anno_match.group(1).split(
                                            ","
                                        )
                                        if input_type_anno_match
                                        else []
                                    )
                                    # If annotation 1 failed, try annotation 2
                                    if dims_anno == []:
                                        # Annotation Style 2: System Injected, information included in  input dims/type
                                        dims_anno = anno.get("args", {}).get(
                                            "Input Dims", "N/A"
                                        )
                                        input_type_anno = anno.get(
                                            "args", {}
                                        ).get("Input type", "N/A")

                                        if dims_anno == "N/A":
                                            # Annotation 2 failed as well, meaning the system injection is empty. try next.
                                            continue
                            else:
                                # Annotation Style 1: User Injected, information included in name
                                name_anno = anno.get("name")
                                dims_anno = re.findall(
                                    r"(\w+=\([^)]+\))", name_anno
                                )
                                dims_anno = [
                                    list(map(int, re.findall(r"\d+", s)))
                                    for s in dims_anno
                                ]

                                input_type_anno_match = re.search(
                                    r"dtype=([^\]]+)\]", name_anno
                                )
                                input_type_anno = (
                                    input_type_anno_match.group(1).split(",")
                                    if input_type_anno_match
                                    else []
                                )

                            break
                    self.individual_info.append(
                        (
                            name,
                            dims_anno,
                            input_type_anno,
                            "",
                            "",
                            duration,
                            "",
                            "",
                            "",
                            call_stack,
                        )
                    )

        return self.individual_info

    def get_aggregate_kernel_info(self) -> list[tuple]:
        """
        Aggregates kernel profiling data by grouping entries with identical name,
        dimensions, and input type.

        Returns:
            list: A list mapping each unique (name, dims, input_type) combination
            to its aggregated profiling data. 
        """
        # Create a dictionary to store folded information
        folded_info = {}
        for (
            name,
            dims,
            input_type,
            io,
            desc,
            duration,
            op,
            operation,
            source_code,
            call_stack,
        ) in self.individual_info:
            key = (name, str(dims), str(input_type))
            if key in folded_info:
                folded_info[key][0] += duration
                folded_info[key][1] += 1
            else:
                folded_info[key] = [duration, 1]

        # Convert back to list format with count
        self.aggregate_kernel_info = [
            (name, dims, input_type, duration, count)
            for (name, dims, input_type), (
                duration,
                count,
            ) in folded_info.items()
        ]
        return self.aggregate_kernel_info

    def get_kernel_e2e_time(self, individual_info: list[tuple]) -> float:
        """
        Calculates the total end-to-end (E2E) duration for the provided
        kernel events.

        Args:
            individual_info (list): A list of tuples, each representing a
            kernel event. The duration for each event is expected at index
            5 of the tuple.

        Returns:
            float: The total E2E duration (in microseconds) for all kernel
            events in `individual_info`.
        """
        kernel_e2e_time = 0
        for kernel in individual_info:
            real_duration = kernel[5]  # Duration in microseconds
            kernel_e2e_time += real_duration

        return kernel_e2e_time

    def save_individual_csv(self, output_dir: str = ".") -> None:
        """
        Writes the contents of `self.individual_info` to a CSV file in the specified
        output directory. 

        Args:
            output_dir (str): Directory where the CSV file will be saved. Defaults to
            the current directory.

        Returns:
            None. The CSV file is created at the specified location, containing one
            row per kernel event.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, f"{self.base_name}.csv")
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
                    "Name",
                    "Dims",
                    "Data Type",
                    "Input/Output",
                    "Descriptions",
                    "Duration (us)",
                    "op",
                    "operation",
                    "Source Code",
                    "Call Stack",
                ]
            )
            csv_writer.writerows(self.individual_info)

import random
from typing import Iterator, List, Optional
from torch.utils.data import Sampler


class DynamicLengthBatchSampler(Sampler):
    def __init__(
        self,
        args,
        seed=None
    ):
        """
        Initialize the dynamic length batch sampler.
        """
        super().__init__(None)

        self.args = args

        # Validate inputs
        if args.min_seq_length > args.max_seq_length:
            raise ValueError("min_seq_length cannot be greater than max_seq_length")
        if args.min_seq_length % 2 != 0 or args.max_seq_length % 2 != 0:
            raise ValueError("min and max sequence lengths must be even numbers")
        if args.step_size % 2 != 0:
            raise ValueError("step_size must be even")
        if not 0 < args.length_similarity_ratio < 1:
            raise ValueError("length_similarity_ratio must be between 0 and 1")
        if not 0 < args.min_batch_utilization <= 1:
            raise ValueError("min_batch_utilization must be between 0 and 1")

        self.rnn_chunk_len = int(getattr(args, "rnn_chunk_len", 500) or 500)
        if self.rnn_chunk_len <= 0:
            raise ValueError("rnn_chunk_len must be a positive integer")

        self.subsample_batchsize = int(getattr(args, "subsample_batchsize", 0) or 0)

        self.max_seq_length = args.max_seq_length
        self.min_seq_length = args.min_seq_length
        self.step_size = args.step_size
        self.max_tokens_per_batch = args.max_tokens_per_batch
        self.length_similarity_ratio = args.length_similarity_ratio
        self.min_batch_utilization = args.min_batch_utilization
        self.max_attempts = args.max_attempts
        self.rng = random.Random(seed)

        # Pre-compute valid sequence lengths
        self.valid_lengths = list(range(
            args.min_seq_length,
            args.max_seq_length + 1,
            args.step_size
        ))
        # Extra safety: keep only even lengths
        self.valid_lengths = [l for l in self.valid_lengths if l % 2 == 0]

    def _num_chunks(self, L: int) -> int:
        """Number of RNN chunks for a sequence of length L."""
        return (int(L) + self.rnn_chunk_len - 1) // self.rnn_chunk_len

    def get_similar_lengths(self, reference_length: int) -> List[int]:
        """
        Get valid lengths that are within the similarity ratio of the reference length.
        """
        max_diff = int(reference_length * self.length_similarity_ratio)
        min_allowed = max(self.min_seq_length, reference_length - max_diff)
        max_allowed = min(self.max_seq_length, reference_length + max_diff)

        return [l for l in self.valid_lengths
                if min_allowed <= l <= max_allowed]

    def build_batch(self, first_length: Optional[int] = None) -> List[int]:
        """
        Build a single batch starting with the given length or choosing one randomly.
        Returns empty list if unable to build a valid batch.

          • All lengths in the batch must yield the same number of chunks
            when divided by rnn_chunk_len (ceil division).
          • After a valid batch is formed, optionally subsample to at most
            subsample_batchsize sequences.
        """
        if first_length is None:
            # Select initial length that allows for good batch utilization
            max_first_length = min(
                self.max_seq_length,
                int(self.max_tokens_per_batch * 0.5)  # Ensure room for multiple sequences
            )
            valid_first_lengths = [l for l in self.valid_lengths if l <= max_first_length]

            if not valid_first_lengths:
                raise ValueError(
                    f"No valid first lengths: max_tokens_per_batch={self.max_tokens_per_batch} "
                    f"is smaller than min_seq_length={self.min_seq_length}"
                )

            first_length = self.rng.choice(valid_first_lengths)

        current_batch = [first_length]
        current_tokens = first_length

        # Equal-chunk K for the whole batch
        target_chunks = self._num_chunks(first_length)

        # Keep adding similar-length sequences until close to max tokens
        while current_tokens < self.max_tokens_per_batch:
            remaining_tokens = self.max_tokens_per_batch - current_tokens

            # Get lengths similar to the first length that fit in remaining tokens
            similar_lengths = self.get_similar_lengths(first_length)
            similar_lengths = [l for l in similar_lengths if self._num_chunks(l) == target_chunks]

            valid_choices = [l for l in similar_lengths if l <= remaining_tokens]

            if not valid_choices:
                break

            # Prioritize lengths that will lead to good utilization
            target_remaining = self.max_tokens_per_batch - current_tokens
            best_choices = [l for l in valid_choices
                            if l >= target_remaining * self.min_batch_utilization]

            if best_choices:
                valid_choices = best_choices

            # Randomly select from valid similar lengths
            choice = self.rng.choice(valid_choices)

            current_batch.append(choice)
            current_tokens += choice

        # Check if batch meets utilization threshold
        utilization = current_tokens / self.max_tokens_per_batch
        if utilization >= self.min_batch_utilization:
            # Cap batch size after forming a valid batch
            if self.subsample_batchsize > 0 and len(current_batch) > self.subsample_batchsize:
                current_batch = self.rng.sample(current_batch, self.subsample_batchsize)
            return current_batch
        return []

    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches of similarly-sized sequence lengths that satisfy the constraints
        and achieve good batch utilization.

        Returns:
            Iterator yielding lists of sequence lengths for each batch
        """
        if self.args.one_sequence_per_iter:
            while True:
                virtual_batch = self.build_batch()
                for length in virtual_batch:
                    yield [length]
        else:
            while True:
                # Try multiple times to build a well-utilized batch
                for _ in range(self.max_attempts):
                    batch = self.build_batch()
                    if batch:  # Found a good batch
                        yield batch
                        break
                else:  # If all attempts failed, relax constraints slightly
                    fallback_utilization = self.min_batch_utilization * 0.9
                    batch = self.build_batch()
                    if batch and sum(batch) >= self.max_tokens_per_batch * fallback_utilization:
                        yield batch
                    else:
                        # If even fallback fails, yield whatever build_batch returns
                        yield self.build_batch()

    def __len__(self) -> int:
        """
        Returns an approximate number of batches.
        Note: Actual number may vary due to dynamic batch sizes.
        """
        return int(1e6)

    @property
    def batch_size(self) -> int:
        return 1

    @staticmethod
    def add_command_line_options(argparser):
        argparser.add_argument(
            '--max_seq_length',
            type=int,
            default=1000,
            help='Maximum allowed sequence length'
        )
        argparser.add_argument(
            '--min_seq_length',
            type=int,
            default=10,
            help='Minimum allowed sequence length'
        )
        argparser.add_argument(
            '--step_size',
            type=int,
            default=2,
            help='Step size between valid sequence lengths'
        )
        argparser.add_argument(
            '--max_tokens_per_batch',
            type=int,
            default=1000,
            help='Maximum total tokens allowed in a batch'
        )
        argparser.add_argument(
            '--length_similarity_ratio',
            type=float,
            default=0.25,
            help='Max allowed deviation from first sequence length (as a ratio of the first length)'
        )
        argparser.add_argument(
            '--min_batch_utilization',
            type=float,
            default=0.85,
            help='Minimum ratio of tokens used vs max_tokens_per_batch'
        )
        argparser.add_argument(
            '--max_attempts',
            type=int,
            default=10,
            help='Maximum attempts to build a well-utilized batch'
        )
        argparser.add_argument(
            '--one_sequence_per_iter',
            action='store_true',
            help='If enabled, only return one sequence instead of a batch per iter'
        )

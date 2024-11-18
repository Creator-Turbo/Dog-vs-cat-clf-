document.getElementById('image').addEventListener('change', function(event) {
                              const preview = document.getElementById('preview');
                              preview.src = URL.createObjectURL(event.target.files[0]);
                              preview.onload = () => URL.revokeObjectURL(preview.src);
                          });
                          
document.getElementById("uploadForm").addEventListener("submit", function (e) {
                              e.preventDefault(); // Prevent the default form submission
                          
                              const formData = new FormData();
                              const fileInput = document.getElementById("fileInput").files[0];
                          
                              if (!fileInput) {
                                  alert("Please upload an image!");
                                  return;
                              }
                          
                              formData.append("file", fileInput);
                          
                              // Show loading message
                              const resultDiv = document.getElementById("result");
                              resultDiv.innerHTML = "<p>Loading...</p>";
                          
                              // Send image to the server via AJAX
                              fetch("/predict", {
                                  method: "POST",
                                  body: formData,
                              })
                                  .then((response) => response.json())
                                  .then((data) => {
                                      if (data.error) {
                                          resultDiv.innerHTML = `<p style="color: red;">${data.error}</p>`;
                                      } else {
                                          resultDiv.innerHTML = `
                                              <img src="${data.image_url}" alt="Uploaded Image" class="uploaded-image">
                                              <h2>It's a ${data.prediction}!</h2>
                                          `;
                                      }
                                  })
                                  .catch((error) => {
                                      console.error("Error:", error);
                                      resultDiv.innerHTML = `<p style="color: red;">An error occurred. Please try again later.</p>`;
                                  });
                          });
                          
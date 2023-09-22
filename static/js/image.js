"use strict";
async function postAction(action, data = {}) {
  const formData = new FormData();
  formData.append("action", action);
  for (const key in data) {
    formData.append(key, data[key]);
  }
  const response = await fetch("/", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.text())
    .then((html) => {
      var newDoc = document.implementation.createHTMLDocument("New Document");
      newDoc.documentElement.innerHTML = html;

      // Replace the current document with the new one
      document.open();
      document.write(newDoc.documentElement.innerHTML);
      document.close();
    });
}
function addErrorMessage(message) {
  $("#messageContainer").append(`
    <div class="alert alert-danger m-1 me-0 d-flex" role="alert">
        ${message}
        <button type="button" class="btn-close ms-auto"  data-bs-dismiss="alert" aria-label="Close"></button>
    </div>
`);
}
function addInfoMessage(message) {
  $("#messageContainer").append(`
    <div class="alert alert-primary m-1 me-0 align d-flex " role="alert">
        ${message}
        <button type="button" class="btn-close ms-auto"  data-bs-dismiss="alert" aria-label="Close"></button>
    </div>
`);
}

function getImgParams($comboBox, areaNumber) {
  const faceNum = parseInt($comboBox.val());
  const fileName = current_images[areaNumber - 1];
  return { faceNum, fileName };
}
async function getFacePath($comboBox, areaNumber) {
  let $params = getImgParams($comboBox, areaNumber);
  let path = "";
  let face_num = $params.faceNum;
  if ($params.faceNum == -2) {
    path = `pool/${$params.fileName}`;
  } else {
    path = `static/aligned_${face_num}_${$params.fileName}`;
  }
  const exists = await checkFileExists(path);
  return { path, exists, face_num };
}
function setupDropArea($dropArea, areaNumber) {
  const $img = $dropArea.find("img");
  const $content = $dropArea.find(`#dragarea${areaNumber}-content`);
  const $dragText = $content.find("header");
  const $button = $content.find("button");
  const $input = $dropArea.find("input");
  const $comboBox = $(`#combo-box${areaNumber}`);

  $button.on("click", function () {
    $input.click();
  });

  $input.on("change", function () {
    const file = this.files[0];
    $dropArea.addClass("active");
    showFile($dropArea, file);
  });
  $dropArea.on("dragover", function (event) {
    event.preventDefault();
    $dropArea.addClass("active");
    $dragText.text("Release to Upload File");
  });
  $dropArea.on("dragleave", function () {
    $dropArea.removeClass("active");
    $dragText.text("Drag & Drop to Upload File");
  });

  $dropArea.on("drop", function (event) {
    event.preventDefault();
    const file = event.originalEvent.dataTransfer.files[0];
    $input[0].files = event.originalEvent.dataTransfer.files;
    showFile($dropArea, file);
  });
  $comboBox.on("change", async function () {
    let result = await getFacePath($(this), areaNumber);
    if (result.exists) {
      //display
      $(`#face_num_input${areaNumber}`).val(result.face_num);
      $img.attr("src", result.path);
      $img.removeClass("d-none");
      $dropArea.removeClass("p-5");
    } else {
      // align and then display
    }
  });
  return { $img, $content, $dragText, $button, $input, $comboBox };
}
const dropArea1Elements = setupDropArea($("#dragarea1"), 1);
const dropArea2Elements = setupDropArea($("#dragarea2"), 2);
let file;

$(document).ready(function () {
  let $imgs = [dropArea1Elements.$img, dropArea2Elements.$img];
  for (let index = 0; index < current_images.length; index++) {
    if (current_images[index] != undefined && current_images[index] !== "") {
      let path1 = `pool/${current_images[index]}`;
      $imgs[index].attr("src", path1);
      $imgs[index].removeClass("d-none");
      $imgs[index].removeClass("p-5");
    }
  }
});

errors.forEach((error) => {
  if (error !== "") {
    addErrorMessage(error);
  }
});
messages.forEach((message) => {
  if (message !== "") {
    addInfoMessage(message);
  }
});
$("#compareBtn").on("click", async function () {
  console.log(current_images);
  let result1 = await getImgParams(dropArea1Elements.$comboBox, 1);
  let result2 = await getImgParams(dropArea2Elements.$comboBox, 2);
  //   if (!result1.exists) {
  //   }
  //   if (!result2.exists) {
  //   }
  //   both exist

  console.log(result1.fileName);
  console.log(result2.fileName);
  await postAction("Compare", {
    image1: result1.fileName,
    face1: result1.faceNum,
    image2: result2.fileName,
    face2: result2.faceNum,
  });
});
function deleteImage(button){
    var $dragArea = $(button).parent();
    unshowFile($dragArea);
    $(button).remove();
}
function unshowFile($dragArea){
    let $img=$dragArea.find("img").attr("src", '');
    $img.addClass("d-none");
    $dragArea.addClass("p-5");
    $dragArea.removeClass("active");

}
function showFile(dragArea, file) {
  let fileType = file.type;
  let validExtensions = ["image/jpeg", "image/jpg", "image/png"];
  if (validExtensions.includes(fileType)) {
    let fileReader = new FileReader();
    fileReader.onload = () => {
      let fileURL = fileReader.result;
      let $imgElement = dragArea.find("img");
      $imgElement.attr("src", fileURL);
      $imgElement.removeClass("d-none");
      dragArea.append(
        `<button type="button" onclick="deleteImage(this)"
        class="btn btn-danger z-3 text-light position-absolute top-0 end-0 rounded-top-1">
        <i class="fas fa-solid fa-trash"></i>
        </button>`
      );
      dragArea.removeClass("p-5");
    };
    fileReader.readAsDataURL(file);
  } else {
    alert("This is not an Image File!");
    dragArea.removeClass("active");
    dragArea.find("header").text("Drag & Drop to Upload File");
  }
}
async function checkFileExists(filePath) {
  try {
    await $.ajax({
      url: filePath,
      type: "HEAD",
    });
    return true;
  } catch (error) {
    return false;
  }
}

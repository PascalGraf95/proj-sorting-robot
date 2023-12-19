#include "NewActorComponent.h"
#include "Engine/Engine.h"
#include "EngineUtils.h"
#include "Engine/GameViewportClient.h"

// Sets default values for this component's properties
UNewActorComponent::UNewActorComponent()
{
    // Set this component to be initialized when the game starts, and to be ticked every frame.
    PrimaryComponentTick.bCanEverTick = true;
}

// Called when the game starts
void UNewActorComponent::BeginPlay()
{
    Super::BeginPlay();
    // ...
}

// Called every frame
void UNewActorComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    // Überprüfen Sie auf Null, bevor Sie auf GEngine und GetGameViewportClient() zugreifen
    if (GEngine && GEngine->GameViewport && GEngine->GameViewport->Viewport)
    {
        // Erfassen Sie das Bild
        TArray<FColor> Bitmap;
        FIntRect CaptureRegion(0, 0, GEngine->GameViewport->Viewport->GetSizeXY().X, GEngine->GameViewport->Viewport->GetSizeXY().Y);
        GetGameViewportClient()->ReadPixels(Bitmap, FReadSurfaceDataFlags(), CaptureRegion);

        // Wandeln Sie die Farbdaten in ein geeignetes Format um (z.B. PNG)
        TArray<uint8> CompressedBitmap;
        FImageUtils::CompressImageArray(CaptureRegion.GetSize().X, CaptureRegion.GetSize().Y, Bitmap, CompressedBitmap);

        // Senden Sie das Bild über TCP
        FSocket* Socket = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateSocket(NAME_Stream, TEXT("default"), false);
        FIPv4Address Addr;
        FIPv4Address::Parse(TEXT("127.0.0.1"), Addr);
        TSharedRef<FInternetAddr> RemoteAddress = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();
        RemoteAddress->SetIp(Addr.GetValue());
        RemoteAddress->SetPort(12345); // Port, den Sie verwenden möchten

        if (Socket->Connect(*RemoteAddress))
        {
            int32 BytesSent = 0;
            Socket->Send(CompressedBitmap.GetData(), CompressedBitmap.Num(), BytesSent);
        }
    }
}

// Funktion zum Erfassen und Senden eines Bildes über TCP
void UNewActorComponent::YourFunctionToCaptureAndSendImage()
{

}
